import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import PageHeader from "../components/PageHeader";
import { analyze, fetchRaces, fetchSeasons, fetchSession } from "../services/api";

const exampleConfig = {
  year: 2024,
  race: "Monaco Grand Prix",
  driver: "VER",
  comparisonDriver: "LEC",
  team1: "Red Bull Racing",
  team2: "Ferrari",
};

export default function AnalysisPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const [seasons, setSeasons] = useState([]);
  const [races, setRaces] = useState([]);
  const [drivers, setDrivers] = useState([]);
  const [teams, setTeams] = useState([]);

  const [year, setYear] = useState(2025);
  const [race, setRace] = useState("");
  const [driver, setDriver] = useState("");
  const [comparisonDriver, setComparisonDriver] = useState("");
  const [team1, setTeam1] = useState("");
  const [team2, setTeam2] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const selectedRaceLabel = useMemo(() => {
    const m = races.find((r) => String(r.round) === String(race) || r.race_name === race);
    return m?.race_name || race;
  }, [race, races]);

  const comparisonOptions = useMemo(
    () => drivers.filter((d) => d.code !== driver),
    [drivers, driver]
  );

  useEffect(() => {
    async function loadSeasons() {
      try {
        const seasonValues = await fetchSeasons();
        setSeasons(seasonValues);
        setYear(seasonValues[seasonValues.length - 1]);
      } catch (e) {
        setError(e?.response?.data?.detail || e.message);
      }
    }
    loadSeasons();
  }, []);

  useEffect(() => {
    if (!year) return;
    async function loadRaces() {
      try {
        setError("");
        const raceList = await fetchRaces(year);
        setRaces(raceList);
        if (raceList.length > 0) {
          setRace(raceList[0].race_name);
        }
      } catch (e) {
        setError(e?.response?.data?.detail || e.message);
      }
    }
    loadRaces();
  }, [year]);

  useEffect(() => {
    if (!year || !race) return;
    async function loadSessionMeta() {
      try {
        setError("");
        const data = await fetchSession(year, race);
        const driverList = data.drivers || [];
        setDrivers(driverList);
        setTeams(data.teams || []);

        if (driverList.length) {
          const primary = driverList[0].code;
          setDriver(primary);
          const compare = driverList.find((d) => d.code !== primary)?.code || "";
          setComparisonDriver(compare);
        }

        if (data.teams?.length >= 2) {
          setTeam1(data.teams[0]);
          setTeam2(data.teams[1]);
        }
      } catch (e) {
        setError(e?.response?.data?.detail || e.message);
      }
    }
    loadSessionMeta();
  }, [year, race]);

  useEffect(() => {
    if (comparisonDriver && comparisonDriver !== driver) return;
    const fallback = comparisonOptions[0]?.code || "";
    setComparisonDriver(fallback);
  }, [driver, comparisonDriver, comparisonOptions]);

  useEffect(() => {
    if (searchParams.get("example") === "1") {
      setYear(exampleConfig.year);
      setRace(exampleConfig.race);
      setDriver(exampleConfig.driver);
      setComparisonDriver(exampleConfig.comparisonDriver);
      setTeam1(exampleConfig.team1);
      setTeam2(exampleConfig.team2);
    }
  }, [searchParams]);

  async function handleAnalyze() {
    if (!year || !race || !driver || !comparisonDriver || !team1 || !team2) {
      setError("Please complete all configuration steps.");
      return;
    }

    if (driver === comparisonDriver) {
      setError("Comparison driver must be different from selected driver.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await analyze({
        year,
        race,
        driver,
        comparison_driver: comparisonDriver,
        team1,
        team2,
      });
      navigate(`/results/${response.job_id}`, {
        state: {
          year,
          race: selectedRaceLabel,
          driver,
          comparisonDriver,
          team1,
          team2,
        },
      });
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen px-6 py-10 md:px-10 lg:px-20">
      <div className="mx-auto max-w-5xl">
        <PageHeader
          title="Analysis Configuration"
          subtitle="Step 1 Select Season. Step 2 Select Race. Step 3 Select Driver. Step 4 Select Comparison Driver. Step 5 Select Teams."
        />

        <section className="grid gap-5 md:grid-cols-2">
          <div className="card p-5">
            <label className="text-sm text-white/70">Step 1: Select Season</label>
            <select value={year} onChange={(e) => setYear(Number(e.target.value))} className="mt-2 w-full rounded-md bg-black/40 p-2">
              {seasons.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          <div className="card p-5">
            <label className="text-sm text-white/70">Step 2: Select Race</label>
            <select value={race} onChange={(e) => setRace(e.target.value)} className="mt-2 w-full rounded-md bg-black/40 p-2">
              {races.map((r) => (
                <option key={`${r.round}-${r.race_name}`} value={r.race_name}>{r.race_name}</option>
              ))}
            </select>
          </div>

          <div className="card p-5">
            <label className="text-sm text-white/70">Step 3: Select Driver</label>
            <select value={driver} onChange={(e) => setDriver(e.target.value)} className="mt-2 w-full rounded-md bg-black/40 p-2">
              {drivers.map((d) => (
                <option key={d.code} value={d.code}>{d.code} ({d.team})</option>
              ))}
            </select>
          </div>

          <div className="card p-5">
            <label className="text-sm text-white/70">Step 4: Select Comparison Driver</label>
            <select
              value={comparisonDriver}
              onChange={(e) => setComparisonDriver(e.target.value)}
              className="mt-2 w-full rounded-md bg-black/40 p-2"
            >
              {comparisonOptions.map((d) => (
                <option key={`cmp-${d.code}`} value={d.code}>{d.code} ({d.team})</option>
              ))}
            </select>
          </div>

          <div className="card p-5 md:col-span-2">
            <label className="text-sm text-white/70">Step 5: Select Teams for Comparison</label>
            <div className="mt-2 grid grid-cols-2 gap-3">
              <select value={team1} onChange={(e) => setTeam1(e.target.value)} className="w-full rounded-md bg-black/40 p-2">
                {teams.map((t) => <option key={`t1-${t}`} value={t}>{t}</option>)}
              </select>
              <select value={team2} onChange={(e) => setTeam2(e.target.value)} className="w-full rounded-md bg-black/40 p-2">
                {teams.map((t) => <option key={`t2-${t}`} value={t}>{t}</option>)}
              </select>
            </div>
          </div>
        </section>

        {error ? <p className="mt-4 text-sm text-red-300">{error}</p> : null}

        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="mt-6 rounded-lg bg-f1red px-8 py-3 font-bold text-white disabled:opacity-60"
        >
          {loading ? "Running Statistical Analysis..." : "Run Statistical Analysis"}
        </button>
      </div>
    </main>
  );
}
