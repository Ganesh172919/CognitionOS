export default function ExecutionTimeline({ timelineData }: any) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-lg font-semibold mb-4">Execution Timeline</h2>
      <div className="space-y-2">
        {timelineData?.events?.map((event: any, i: number) => (
          <div key={i} className="flex items-center gap-4 p-2 hover:bg-gray-50 rounded">
            <span className="text-sm text-gray-600">{event.name}</span>
            <span className="text-xs text-gray-500">{event.duration_ms}ms</span>
          </div>
        ))}
      </div>
    </div>
  );
}
