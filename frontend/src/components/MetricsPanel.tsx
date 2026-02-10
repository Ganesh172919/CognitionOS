export default function MetricsPanel({ dashboardData }: any) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-lg font-semibold mb-4">Service Metrics</h2>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {dashboardData?.service_metrics && Object.entries(dashboardData.service_metrics).map(([service, metrics]: [string, any]) => (
          <div key={service} className="p-4 border border-gray-200 rounded-lg">
            <h3 className="font-medium text-gray-900 text-sm">{service}</h3>
            <p className="text-2xl font-bold text-gray-900 mt-2">{metrics.count}</p>
            <p className="text-xs text-gray-600 mt-1">requests</p>
          </div>
        ))}
      </div>
    </div>
  );
}
