export default function PageHeader({ title, subtitle }) {
  return (
    <header className="mb-6">
      <h1 className="text-3xl md:text-4xl font-extrabold text-white">{title}</h1>
      {subtitle ? <p className="mt-2 text-white/70 max-w-4xl">{subtitle}</p> : null}
    </header>
  );
}
