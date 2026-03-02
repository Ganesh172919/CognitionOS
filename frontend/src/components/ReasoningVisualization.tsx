/**
 * Reasoning Visualization Component
 *
 * Displays agent reasoning steps with confidence scores and decision points.
 * Premium dark theme with glow effects and animations.
 */

import { Brain, CheckCircle2, AlertCircle, ChevronDown, ChevronUp, Eye } from 'lucide-react';
import { useState } from 'react';

interface ReasoningStep {
  step: number;
  type: string;
  description: string;
  confidence?: number;
  alternatives?: any[];
  rationale?: string;
}

interface ReasoningData {
  total_steps: number;
  reasoning_summary: string;
  key_decisions: ReasoningStep[];
  confidence_scores: Record<string, number>;
  overall_quality: number;
}

export default function ReasoningVisualization({ reasoningData }: { reasoningData: ReasoningData }) {
  const [expandedStep, setExpandedStep] = useState<number | null>(null);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-emerald-400';
    if (confidence >= 0.7) return 'text-amber-400';
    return 'text-red-400';
  };

  const getConfidenceBarColor = (score: number) => {
    if (score >= 0.8) return 'bg-emerald-500';
    if (score >= 0.6) return 'bg-amber-500';
    return 'bg-red-500';
  };

  return (
    <div className="premium-card">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 flex items-center justify-center shadow-glow-brand">
          <Brain className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">Reasoning Trace</h2>
          <p className="text-xs text-zinc-400">Agent decision pathway analysis</p>
        </div>
      </div>

      {/* Summary Card */}
      <div className="mb-6 p-4 rounded-xl bg-surface-3/50 border border-zinc-800">
        <p className="text-sm text-zinc-300 leading-relaxed">{reasoningData.reasoning_summary}</p>
        <div className="mt-4 flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">Steps</span>
            <span className="text-sm font-bold text-white">{reasoningData.total_steps}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">Quality</span>
            <span className={`text-sm font-bold ${getConfidenceColor(reasoningData.overall_quality)}`}>
              {(reasoningData.overall_quality * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Key Decisions */}
      {reasoningData.key_decisions && reasoningData.key_decisions.length > 0 && (
        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-semibold text-zinc-300 flex items-center gap-2">
            <Eye className="w-4 h-4 text-zinc-400" />
            Key Decision Points
          </h3>
          {reasoningData.key_decisions.map((decision, index) => (
            <div
              key={index}
              className="glass-card p-4 cursor-pointer"
              onClick={() => setExpandedStep(expandedStep === index ? null : index)}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-3">
                  <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-brand-500/15 text-brand-400 text-sm font-bold border border-brand-500/20">
                    {decision.step}
                  </span>
                  <div>
                    <span className="text-sm font-medium text-white capitalize">{decision.type}</span>
                    <p className="text-xs text-zinc-400 mt-0.5">{decision.description}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  {decision.confidence !== undefined && (
                    <div className="flex items-center gap-1.5">
                      {decision.confidence >= 0.7 ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-amber-400" />
                      )}
                      <span className={`text-sm font-bold ${getConfidenceColor(decision.confidence)}`}>
                        {(decision.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}
                  {expandedStep === index ? (
                    <ChevronUp className="w-4 h-4 text-zinc-500" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-zinc-500" />
                  )}
                </div>
              </div>

              {/* Expanded Content */}
              {expandedStep === index && (
                <div className="mt-3 pt-3 border-t border-zinc-800 space-y-3 animate-fade-in">
                  {decision.rationale && (
                    <div>
                      <p className="text-xs text-zinc-500 font-medium mb-1">Rationale</p>
                      <p className="text-sm text-zinc-300 italic">{decision.rationale}</p>
                    </div>
                  )}
                  {decision.alternatives && decision.alternatives.length > 0 && (
                    <div>
                      <p className="text-xs text-zinc-500 font-medium mb-1">
                        Evaluated {decision.alternatives.length} alternatives
                      </p>
                      <div className="flex gap-2">
                        {decision.alternatives.map((alt: any, i: number) => (
                          <span key={i} className="text-xs px-2 py-1 rounded-md bg-surface-4 text-zinc-400 border border-zinc-700">
                            {alt.description || `Option ${i + 1}`}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Confidence by Phase */}
      {reasoningData.confidence_scores && Object.keys(reasoningData.confidence_scores).length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-zinc-300 mb-4">Confidence by Phase</h3>
          <div className="space-y-3">
            {Object.entries(reasoningData.confidence_scores).map(([stepType, score]) => (
              <div key={stepType}>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm text-zinc-400 capitalize">{stepType}</span>
                  <span className={`text-sm font-bold ${getConfidenceColor(score)}`}>
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-surface-3 rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-2 rounded-full transition-all duration-700 ease-out ${getConfidenceBarColor(score)}`}
                    style={{ width: `${score * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
