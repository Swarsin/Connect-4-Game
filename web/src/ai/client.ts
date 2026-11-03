import type { AiRequest, AiResponse } from "./worker";

export type { Analysis } from "./analysis";
export type { AiRequest, AiResponse } from "./worker";

/** Promise-based wrapper around the AI Web Worker. */
export class AiClient {
  private worker: Worker;
  private seq = 0;
  private pending = new Map<
    number,
    { resolve: (r: AiResponse) => void; reject: (e: Error) => void }
  >();

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (event: MessageEvent<AiResponse>) => {
      const res = event.data;
      const p = this.pending.get(res.id);
      if (!p) return;
      this.pending.delete(res.id);
      p.resolve(res);
    };
    this.worker.onerror = (event) => {
      const err = new Error(event.message || "AI worker crashed");
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
  }

  request(req: Omit<AiRequest, "id">): Promise<AiResponse> {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...req, id });
    });
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}
