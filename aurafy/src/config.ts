// src/config.ts
export const API = import.meta.env.DEV ? (import.meta.env.VITE_API_BASE ?? "") : "";
