export default function StatCard({ title, value, subtitle }) {
  return (
    <div className="card p-4">
      <p className="text-xs uppercase tracking-wider text-white/60">{title}</p>
      <p className="mt-2 text-2xl font-bold text-white">{value}</p>
      {subtitle ? <p className="mt-1 text-sm text-white/60">{subtitle}</p> : null}
    </div>
  );
}
