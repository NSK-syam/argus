// Minimal SSE parser built on fetch + ReadableStream, since the backend
// sends named events ("event: attempt\ndata: {...}\n\n") and the browser's
// EventSource API can't set a JSON Accept header or be torn down as
// cleanly inside a React effect as a fetch-based reader can.

export interface SseHandlers {
  onEvent: (event: string, data: string) => void;
  onError?: (err: unknown) => void;
}

export function streamSse(url: string, handlers: SseHandlers): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(url, { signal: controller.signal });
      if (!res.body) throw new Error("no response body (SSE not supported?)");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sepIndex: number;
        while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
          const rawEvent = buffer.slice(0, sepIndex);
          buffer = buffer.slice(sepIndex + 2);

          let event = "message";
          const dataLines: string[] = [];
          for (const line of rawEvent.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
          }
          if (dataLines.length) handlers.onEvent(event, dataLines.join("\n"));
        }
      }
    } catch (err) {
      if ((err as { name?: string })?.name !== "AbortError") {
        handlers.onError?.(err);
      }
    }
  })();

  return () => controller.abort();
}
