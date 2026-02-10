/**
 * Reasoning Visualization Component
 *
 * Displays agent reasoning steps with confidence scores and decision points.
 */

import { Brain, CheckCircle2, AlertCircle } from 'lucide-react';

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
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-6">
        <Brain className="w-6 h-6 text-primary-600" />
        <h2 className="text-lg font-semibold text-gray-900">Reasoning Trace</h2>
      </div>

      {/* Summary */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <p className="text-gray-700">{reasoningData.reasoning_summary}</p>
        <div className="mt-4 flex items-center gap-6">
          <div>
            <span className="text-sm text-gray-600">Total Steps: </span>
            <span className="font-semibold text-gray-900">{reasoningData.total_steps}</span>
          </div>
          <div>
            <span className="text-sm text-gray-600">Quality: </span>
            <span className="font-semibold text-gray-900">
              {(reasoningData.overall_quality * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Key Decisions */}
      {reasoningData.key_decisions && reasoningData.key_decisions.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-medium text-gray-900">Key Decision Points</h3>
          {reasoningData.key_decisions.map((decision, index) => (
            <div key={index} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-primary-100 text-primary-700 text-sm font-medium">
                    {decision.step}
                  </span>
                  <span className="font-medium text-gray-900 capitalize">{decision.type}</span>
                </div>
                {decision.confidence !== undefined && (
                  <div className="flex items-center gap-2">
                    {decision.confidence >= 0.7 ? (
                      <CheckCircle2 className="w-4 h-4 text-green-600" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-yellow-600" />
                    )}
                    <span className="text-sm font-medium text-gray-700">
                      {(decision.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                )}
              </div>
              <p className="text-sm text-gray-700 mb-2">{decision.description}</p>
              {decision.rationale && (
                <p className="text-xs text-gray-600 italic mt-2">
                  <strong>Rationale:</strong> {decision.rationale}
                </p>
              )}
              {decision.alternatives && decision.alternatives.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <p className="text-xs text-gray-600 mb-2">
                    Evaluated {decision.alternatives.length} alternatives
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Confidence by Step Type */}
      {reasoningData.confidence_scores && Object.keys(reasoningData.confidence_scores).length > 0 && (
        <div className="mt-6">
          <h3 className="font-medium text-gray-900 mb-4">Confidence by Phase</h3>
          <div className="space-y-3">
            {Object.entries(reasoningData.confidence_scores).map(([stepType, score]) => (
              <div key={stepType}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-gray-700 capitalize">{stepType}</span>
                  <span className="text-sm font-medium text-gray-900">
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      score >= 0.8
                        ? 'bg-green-600'
                        : score >= 0.6
                        ? 'bg-yellow-600'
                        : 'bg-red-600'
                    }`}
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
