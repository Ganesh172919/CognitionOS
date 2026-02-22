"""
Real-Time Collaboration Engine — Enables multiple users to co-edit documents,
code, and workflows simultaneously with operational transformation (OT),
presence awareness, conflict resolution, and version history.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────── Enums ───────────────────────────────────


class DocumentType(str, Enum):
    TEXT = "text"
    CODE = "code"
    WORKFLOW = "workflow"
    SPREADSHEET = "spreadsheet"
    CANVAS = "canvas"
    JSON = "json"


class OperationType(str, Enum):
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    MOVE = "move"
    FORMAT = "format"
    META_UPDATE = "meta_update"


class ConflictStrategy(str, Enum):
    OT = "operational_transform"
    CRDT = "crdt"
    LAST_WRITER_WINS = "last_writer_wins"
    MERGE = "merge"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    ARCHIVED = "archived"


class PresenceStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    AWAY = "away"
    OFFLINE = "offline"


# ────────────────────────────── Data structures ──────────────────────────────


@dataclass
class Operation:
    op_id: str
    op_type: OperationType
    author_id: str
    document_id: str
    position: int
    content: Any
    length: Optional[int]
    revision: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "op_type": self.op_type.value,
            "author_id": self.author_id,
            "document_id": self.document_id,
            "position": self.position,
            "content": self.content,
            "length": self.length,
            "revision": self.revision,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Cursor:
    user_id: str
    document_id: str
    position: int
    selection_start: Optional[int]
    selection_end: Optional[int]
    color: str
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "document_id": self.document_id,
            "position": self.position,
            "selection_start": self.selection_start,
            "selection_end": self.selection_end,
            "color": self.color,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Presence:
    user_id: str
    display_name: str
    session_id: str
    status: PresenceStatus
    current_document: Optional[str]
    cursor: Optional[Cursor]
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "session_id": self.session_id,
            "status": self.status.value,
            "current_document": self.current_document,
            "cursor": self.cursor.to_dict() if self.cursor else None,
            "joined_at": self.joined_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CollaborativeDocument:
    doc_id: str
    title: str
    doc_type: DocumentType
    content: Any
    revision: int
    owner_id: str
    collaborators: List[str]
    operation_log: List[str]
    conflict_strategy: ConflictStrategy
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type.value,
            "content": self.content,
            "revision": self.revision,
            "owner_id": self.owner_id,
            "collaborators": self.collaborators,
            "operation_count": len(self.operation_log),
            "conflict_strategy": self.conflict_strategy.value,
            "content_hash": self.compute_hash(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CollaborationSession:
    session_id: str
    document_id: str
    participants: List[str]
    status: SessionStatus
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "document_id": self.document_id,
            "participants": self.participants,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "participant_count": len(self.participants),
            "metadata": self.metadata,
        }


# ─────────────────────── Operational Transform Engine ────────────────────────


class OperationalTransformEngine:
    """
    Implements operational transformation to resolve concurrent editing
    conflicts by transforming operations against each other.
    """

    def transform(
        self,
        op_a: Operation,
        op_b: Operation,
    ) -> Tuple[Operation, Operation]:
        """
        Returns transformed (op_a', op_b') such that applying op_a then op_a'
        and applying op_b then op_b' yield the same document state.
        """
        a_prime = self._transform_single(op_a, op_b)
        b_prime = self._transform_single(op_b, op_a)
        return a_prime, b_prime

    def compose(self, op_a: Operation, op_b: Operation) -> Operation:
        """Compose two sequential operations into a single operation."""
        if op_a.op_type == OperationType.INSERT and op_b.op_type == OperationType.INSERT:
            if op_b.position >= op_a.position + len(str(op_a.content)):
                # Sequential inserts can be merged if adjacent
                return Operation(
                    op_id=str(uuid.uuid4()),
                    op_type=OperationType.INSERT,
                    author_id=op_a.author_id,
                    document_id=op_a.document_id,
                    position=op_a.position,
                    content=str(op_a.content) + str(op_b.content),
                    length=None,
                    revision=max(op_a.revision, op_b.revision),
                )
        return op_b  # Default: latter operation wins

    def _transform_single(
        self, op_to_transform: Operation, concurrent_op: Operation
    ) -> Operation:
        new_position = op_to_transform.position
        new_content = op_to_transform.content

        if concurrent_op.op_type == OperationType.INSERT:
            insert_len = len(str(concurrent_op.content))
            if concurrent_op.position <= op_to_transform.position:
                new_position = op_to_transform.position + insert_len

        elif concurrent_op.op_type == OperationType.DELETE:
            delete_len = concurrent_op.length or 1
            if concurrent_op.position < op_to_transform.position:
                new_position = max(
                    concurrent_op.position,
                    op_to_transform.position - delete_len,
                )

        return Operation(
            op_id=str(uuid.uuid4()),
            op_type=op_to_transform.op_type,
            author_id=op_to_transform.author_id,
            document_id=op_to_transform.document_id,
            position=new_position,
            content=new_content,
            length=op_to_transform.length,
            revision=op_to_transform.revision,
            metadata={**op_to_transform.metadata, "transformed": True},
        )


# ─────────────────────── CRDT (Conflict-free Replicated Data Type) ───────────


class CRDTTextDocument:
    """
    Implements a simple sequence CRDT for text documents.
    Each character is assigned a globally unique position ID ensuring
    convergent merges without coordination.
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        # List of (unique_id, char, deleted) tuples
        self._elements: List[Tuple[str, str, bool]] = []
        self._tombstones: Dict[str, bool] = {}

    def insert(self, site_id: str, position: int, text: str) -> List[str]:
        """Insert text at logical position. Returns list of element IDs created."""
        ids_created = []
        actual_position = self._logical_to_actual(position)
        for i, char in enumerate(text):
            elem_id = f"{site_id}:{time.monotonic_ns()}:{i}"
            self._elements.insert(actual_position + i, (elem_id, char, False))
            ids_created.append(elem_id)
        return ids_created

    def delete(self, element_ids: List[str]) -> int:
        """Mark elements as deleted (tombstones). Returns count deleted."""
        count = 0
        for elem_id in element_ids:
            for i, (eid, char, deleted) in enumerate(self._elements):
                if eid == elem_id and not deleted:
                    self._elements[i] = (eid, char, True)
                    self._tombstones[eid] = True
                    count += 1
                    break
        return count

    def get_text(self) -> str:
        return "".join(char for _, char, deleted in self._elements if not deleted)

    def get_length(self) -> int:
        return sum(1 for _, _, deleted in self._elements if not deleted)

    def _logical_to_actual(self, logical_pos: int) -> int:
        actual = 0
        logical = 0
        for eid, char, deleted in self._elements:
            if logical == logical_pos:
                return actual
            if not deleted:
                logical += 1
            actual += 1
        return actual

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._elements)
        deleted = sum(1 for _, _, d in self._elements if d)
        return {
            "doc_id": self.doc_id,
            "total_elements": total,
            "active_elements": total - deleted,
            "tombstones": deleted,
            "text_length": self.get_length(),
        }


# ─────────────────────── Presence Manager ───────────────────────────────────


class PresenceManager:
    """
    Manages user presence, cursor positions, and awareness across collaborative
    sessions with heartbeat-based online detection.
    """

    PRESENCE_TIMEOUT_S = 30.0
    CURSOR_COLORS = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    ]

    def __init__(self):
        self._presences: Dict[str, Presence] = {}
        self._color_assignments: Dict[str, str] = {}
        self._document_users: Dict[str, Set[str]] = defaultdict(set)
        self._session_users: Dict[str, Set[str]] = defaultdict(set)

    def join(
        self,
        user_id: str,
        display_name: str,
        session_id: str,
        document_id: Optional[str] = None,
    ) -> Presence:
        color = self._assign_color(user_id)
        presence = Presence(
            user_id=user_id,
            display_name=display_name,
            session_id=session_id,
            status=PresenceStatus.ACTIVE,
            current_document=document_id,
            cursor=None,
            metadata={"color": color},
        )
        self._presences[user_id] = presence
        self._session_users[session_id].add(user_id)
        if document_id:
            self._document_users[document_id].add(user_id)
        return presence

    def leave(self, user_id: str) -> bool:
        presence = self._presences.pop(user_id, None)
        if presence is None:
            return False
        self._session_users[presence.session_id].discard(user_id)
        if presence.current_document:
            self._document_users[presence.current_document].discard(user_id)
        return True

    def update_cursor(
        self,
        user_id: str,
        document_id: str,
        position: int,
        selection_start: Optional[int] = None,
        selection_end: Optional[int] = None,
    ) -> bool:
        presence = self._presences.get(user_id)
        if presence is None:
            return False
        color = presence.metadata.get("color", "#FF6B6B")
        presence.cursor = Cursor(
            user_id=user_id,
            document_id=document_id,
            position=position,
            selection_start=selection_start,
            selection_end=selection_end,
            color=color,
        )
        presence.last_seen = datetime.now(timezone.utc)
        return True

    def heartbeat(self, user_id: str) -> bool:
        presence = self._presences.get(user_id)
        if presence is None:
            return False
        presence.last_seen = datetime.now(timezone.utc)
        presence.status = PresenceStatus.ACTIVE
        return True

    def get_document_users(self, document_id: str) -> List[Presence]:
        user_ids = self._document_users.get(document_id, set())
        return [
            self._presences[uid]
            for uid in user_ids
            if uid in self._presences
        ]

    def get_session_users(self, session_id: str) -> List[Presence]:
        user_ids = self._session_users.get(session_id, set())
        return [
            self._presences[uid]
            for uid in user_ids
            if uid in self._presences
        ]

    def _assign_color(self, user_id: str) -> str:
        if user_id not in self._color_assignments:
            idx = len(self._color_assignments) % len(self.CURSOR_COLORS)
            self._color_assignments[user_id] = self.CURSOR_COLORS[idx]
        return self._color_assignments[user_id]

    def get_presence_stats(self) -> Dict[str, Any]:
        presences = list(self._presences.values())
        by_status: Dict[str, int] = defaultdict(int)
        for p in presences:
            by_status[p.status.value] += 1
        return {
            "total_online": len(presences),
            "by_status": dict(by_status),
            "active_documents": len(self._document_users),
            "active_sessions": len(self._session_users),
        }


# ─────────────────────── Document Version Control ───────────────────────────


class DocumentVersionControl:
    """
    Tracks document revision history with branching, snapshots,
    and diff computation.
    """

    def __init__(self, max_history_per_doc: int = 100):
        self.max_history = max_history_per_doc
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._snapshots: Dict[str, Dict[int, Any]] = defaultdict(dict)

    def record_revision(
        self,
        document: CollaborativeDocument,
        operation: Operation,
    ) -> int:
        doc_history = self._history[document.doc_id]
        revision_entry = {
            "revision": document.revision,
            "operation": operation.to_dict(),
            "content_hash": document.compute_hash(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        doc_history.append(revision_entry)
        if len(doc_history) > self.max_history:
            self._history[document.doc_id] = doc_history[-self.max_history:]
        return document.revision

    def create_snapshot(
        self, document: CollaborativeDocument, label: str = ""
    ) -> Dict[str, Any]:
        snapshot = {
            "snapshot_id": str(uuid.uuid4()),
            "revision": document.revision,
            "content": document.content,
            "content_hash": document.compute_hash(),
            "label": label,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._snapshots[document.doc_id][document.revision] = snapshot
        return snapshot

    def get_history(
        self, doc_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        return self._history[doc_id][-limit:]

    def get_snapshots(self, doc_id: str) -> List[Dict[str, Any]]:
        return sorted(
            self._snapshots[doc_id].values(),
            key=lambda s: s["revision"],
        )

    def compute_diff(
        self,
        old_content: str,
        new_content: str,
    ) -> Dict[str, Any]:
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        added = [l for l in new_lines if l not in old_lines]
        removed = [l for l in old_lines if l not in new_lines]
        return {
            "lines_added": len(added),
            "lines_removed": len(removed),
            "lines_unchanged": len(old_lines) - len(removed),
            "char_delta": len(new_content) - len(old_content),
            "sample_additions": added[:5],
            "sample_removals": removed[:5],
        }

    def get_version_stats(self) -> Dict[str, Any]:
        return {
            "total_documents_tracked": len(self._history),
            "total_revisions": sum(len(h) for h in self._history.values()),
            "total_snapshots": sum(len(s) for s in self._snapshots.values()),
        }


# ────────────────────── Collaboration Engine ────────────────────────────────


class CollaborationEngine:
    """
    Master collaboration engine integrating document management, real-time
    operations, presence, conflict resolution, and version control.
    """

    def __init__(self):
        self.ot_engine = OperationalTransformEngine()
        self.presence_mgr = PresenceManager()
        self.version_ctrl = DocumentVersionControl()
        self._documents: Dict[str, CollaborativeDocument] = {}
        self._sessions: Dict[str, CollaborationSession] = {}
        self._crdt_docs: Dict[str, CRDTTextDocument] = {}
        self._operation_log: Dict[str, List[Operation]] = defaultdict(list)
        self._change_callbacks: List[Callable] = []

    def create_document(
        self,
        title: str,
        doc_type: DocumentType,
        owner_id: str,
        initial_content: Any = "",
        conflict_strategy: ConflictStrategy = ConflictStrategy.OT,
    ) -> CollaborativeDocument:
        doc_id = str(uuid.uuid4())
        doc = CollaborativeDocument(
            doc_id=doc_id,
            title=title,
            doc_type=doc_type,
            content=initial_content,
            revision=0,
            owner_id=owner_id,
            collaborators=[owner_id],
            operation_log=[],
            conflict_strategy=conflict_strategy,
        )
        self._documents[doc_id] = doc
        if conflict_strategy == ConflictStrategy.CRDT:
            self._crdt_docs[doc_id] = CRDTTextDocument(doc_id)
        return doc

    def create_session(
        self, document_id: str, creator_id: str
    ) -> Optional[CollaborationSession]:
        if document_id not in self._documents:
            return None
        session = CollaborationSession(
            session_id=str(uuid.uuid4()),
            document_id=document_id,
            participants=[creator_id],
            status=SessionStatus.ACTIVE,
        )
        self._sessions[session.session_id] = session
        return session

    def join_session(
        self,
        session_id: str,
        user_id: str,
        display_name: str,
    ) -> Dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            return {"success": False, "error": "Session not found"}
        if session.status != SessionStatus.ACTIVE:
            return {"success": False, "error": f"Session is {session.status.value}"}
        if user_id not in session.participants:
            session.participants.append(user_id)
        doc = self._documents.get(session.document_id)
        if doc and user_id not in doc.collaborators:
            doc.collaborators.append(user_id)
        presence = self.presence_mgr.join(
            user_id, display_name, session_id, session.document_id
        )
        return {
            "success": True,
            "session": session.to_dict(),
            "document": doc.to_dict() if doc else None,
            "presence": presence.to_dict(),
        }

    def apply_operation(
        self,
        doc_id: str,
        user_id: str,
        op_type: OperationType,
        position: int,
        content: Any,
        length: Optional[int] = None,
        client_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        doc = self._documents.get(doc_id)
        if doc is None:
            return {"success": False, "error": "Document not found"}

        operation = Operation(
            op_id=str(uuid.uuid4()),
            op_type=op_type,
            author_id=user_id,
            document_id=doc_id,
            position=position,
            content=content,
            length=length,
            revision=doc.revision,
        )

        # Handle OT if client is behind server revision
        concurrent_ops: List[Operation] = []
        if client_revision is not None and client_revision < doc.revision:
            server_ops = self._operation_log[doc_id]
            concurrent_ops = [
                op
                for op in server_ops
                if op.revision > client_revision
            ]
            for concurrent in concurrent_ops:
                operation, _ = self.ot_engine.transform(operation, concurrent)

        # Apply to document content
        transformed_content = self._apply_to_content(doc, operation)
        doc.content = transformed_content
        doc.revision += 1
        doc.updated_at = datetime.now(timezone.utc)
        doc.operation_log.append(operation.op_id)

        # Store in log
        self._operation_log[doc_id].append(operation)
        if len(self._operation_log[doc_id]) > 500:
            self._operation_log[doc_id] = self._operation_log[doc_id][-500:]

        # Record in version control
        self.version_ctrl.record_revision(doc, operation)

        # Update CRDT if applicable
        if doc.conflict_strategy == ConflictStrategy.CRDT and doc_id in self._crdt_docs:
            crdt_doc = self._crdt_docs[doc_id]
            if op_type == OperationType.INSERT:
                crdt_doc.insert(user_id, position, str(content))
            elif op_type == OperationType.DELETE and length:
                # Get element IDs to delete (simplified)
                pass

        return {
            "success": True,
            "op_id": operation.op_id,
            "new_revision": doc.revision,
            "transforms_applied": len(concurrent_ops),
        }

    def _apply_to_content(
        self, doc: CollaborativeDocument, op: Operation
    ) -> Any:
        content = doc.content
        if doc.doc_type in (DocumentType.TEXT, DocumentType.CODE):
            if not isinstance(content, str):
                content = str(content)
            if op.op_type == OperationType.INSERT:
                pos = min(op.position, len(content))
                content = content[:pos] + str(op.content) + content[pos:]
            elif op.op_type == OperationType.DELETE:
                length = op.length or 1
                pos = min(op.position, len(content))
                content = content[:pos] + content[pos + length:]
            elif op.op_type == OperationType.REPLACE:
                length = op.length or len(str(op.content))
                pos = min(op.position, len(content))
                content = content[:pos] + str(op.content) + content[pos + length:]
        elif doc.doc_type == DocumentType.JSON:
            if isinstance(content, dict) and isinstance(op.content, dict):
                if op.op_type == OperationType.META_UPDATE:
                    content.update(op.content)
        return content

    def update_cursor(
        self,
        user_id: str,
        doc_id: str,
        position: int,
        selection_start: Optional[int] = None,
        selection_end: Optional[int] = None,
    ) -> bool:
        return self.presence_mgr.update_cursor(
            user_id, doc_id, position, selection_start, selection_end
        )

    def get_document(self, doc_id: str) -> Optional[CollaborativeDocument]:
        return self._documents.get(doc_id)

    def get_document_presence(self, doc_id: str) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.presence_mgr.get_document_users(doc_id)]

    def create_snapshot(self, doc_id: str, label: str = "") -> Optional[Dict[str, Any]]:
        doc = self._documents.get(doc_id)
        if doc is None:
            return None
        return self.version_ctrl.create_snapshot(doc, label)

    def get_history(self, doc_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        return self.version_ctrl.get_history(doc_id, limit)

    def list_documents(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        docs = list(self._documents.values())
        if owner_id:
            docs = [d for d in docs if d.owner_id == owner_id or owner_id in d.collaborators]
        return [d.to_dict() for d in docs]

    def list_sessions(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sessions = list(self._sessions.values())
        if doc_id:
            sessions = [s for s in sessions if s.document_id == doc_id]
        return [s.to_dict() for s in sessions]

    def get_engine_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self._documents),
            "total_sessions": len(self._sessions),
            "total_operations": sum(
                len(ops) for ops in self._operation_log.values()
            ),
            "presence": self.presence_mgr.get_presence_stats(),
            "version_control": self.version_ctrl.get_version_stats(),
        }
