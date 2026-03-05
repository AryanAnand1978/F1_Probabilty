import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
});

export async function fetchSeasons() {
  const { data } = await client.get("/seasons");
  return data.seasons || [];
}

export async function fetchRaces(year) {
  const { data } = await client.get(`/races/${year}`);
  return data.races || [];
}

export async function fetchSession(year, race) {
  const { data } = await client.get(`/session/${year}/${encodeURIComponent(race)}`);
  return data;
}

export async function analyze(payload) {
  const { data } = await client.post("/analyze", payload);
  return data;
}

export async function fetchResults(jobId) {
  const { data } = await client.get(`/results/${jobId}`);
  return data;
}
