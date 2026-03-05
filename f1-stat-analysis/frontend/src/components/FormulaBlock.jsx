export default function FormulaBlock({ title, formula, note }) {
  return (
    <div className="card p-4">
      <p className="text-sm font-semibold text-f1red">{title}</p>
      <p className="mt-2 font-mono text-sm text-white break-words">{formula}</p>
      {note ? <p className="mt-2 text-xs text-white/65">{note}</p> : null}
    </div>
  );
}
