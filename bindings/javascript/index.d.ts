export interface CortextConfig {
  focus?: number;
  sensitivity?: number;
  stability?: number;
  affectInterrupt?: boolean;
  affectRetrieval?: boolean;
  reinforcementEnabled?: boolean;
  proceduralEnabled?: boolean;
  sequentialEdgesEnabled?: boolean;
  labelBankPath?: string;
  /**
   * Inference-provider URI for the Summarizer role, e.g.
   * "ollama://127.0.0.1:11435/gemma4:e2b". Omitted/empty keeps local
   * model auto-discovery; an unresolvable URI makes construction throw.
   */
  summarizerProviderUri?: string;
  /** Inference-provider URI for the Extractor role; same semantics. */
  extractorProviderUri?: string;
}

export type CortextContext = Record<string, unknown>;
export interface CortextEmbedding {
  embedding: number[];
  dimension: number;
}

export declare const CONSOLIDATE_SHALLOW: number;
export declare const CONSOLIDATE_DEEP: number;
export declare const CONSOLIDATE_BOTH: number;

export declare class Cortext {
  constructor(config?: CortextConfig, dbPath?: string, modelsDir?: string);
  processTextJson(text: string, sourceId: string): string;
  processText(text: string, sourceId: string): CortextContext;
  embedTextJson(text: string): string;
  processAudioJson(pcm: Float32Array, sourceId: string): string;
  processAudio(pcm: Float32Array, sourceId: string): CortextContext;
  embedAudioJson(pcm: Float32Array): string;
  processImageJson(
    data: Uint8Array,
    width: number,
    height: number,
    channels: number,
    sourceId: string
  ): string;
  processImage(
    data: Uint8Array,
    width: number,
    height: number,
    channels: number,
    sourceId: string
  ): CortextContext;
  embedImageJson(
    data: Uint8Array,
    width: number,
    height: number,
    channels: number
  ): string;
  consolidateJson(): string;
  consolidate(): CortextContext;
  consolidateModeJson(mode: number): string;
  consolidateMode(mode: number): CortextContext;
  flush(): void;
  reset(): void;
}

export declare function version(): string;
export declare function lastError(): string;
