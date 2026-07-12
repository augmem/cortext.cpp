export interface CortextConfig {
  focus?: number;
  sensitivity?: number;
  stability?: number;
  affectInterrupt?: boolean;
  affectRetrieval?: boolean;
  reinforcementEnabled?: boolean;
  proceduralEnabled?: boolean;
  sequentialEdgesEnabled?: boolean;
  signalFilterAudioEnabled?: boolean;
  signalFilterImageEnabled?: boolean;
  signalFilterTextEnabled?: boolean;
}

export interface CortextMemory {
  id?: number | string;
  memory_id?: number | string;
  text?: string;
  source_id?: string;
  timestamp?: number;
  modality?: string;
  mimetype?: string;
  rel?: number;
  salience?: number;
  contradiction?: number;
  usage_count?: number;
  soft_anchors?: unknown[];
  [key: string]: unknown;
}

export interface CortextContext {
  retrieved_memory?: CortextMemory[];
  working_memory?: CortextMemory[];
  should_interrupt?: boolean;
  interrupt_aborted?: boolean;
  at_boundary?: boolean;
  consolidation_recommended?: boolean;
  consolidation_required?: boolean;
  output?: Record<string, unknown>;
  embedding?: number[];
  embedding_dimension?: number;
  encode_ms?: number;
  process_ms?: number;
  hydrate_ms?: number;
  total_ms?: number;
  [key: string]: unknown;
}
export interface CortextEmbedding {
  embedding: number[];
  dimension: number;
}

export interface Media {
  data: Uint8Array;
  mimetype: string;
}

/** Matches cortext::Retention / C API cortext_retention. */
export type Retention =
  | "natural"
  | "durable"
  | "boundary"
  | "ephemeral"
  | 0
  | 1
  | 2
  | 3;

export interface ProcessOptions {
  includeEmbedding?: boolean;
  omitEmbedding?: boolean;
  /** Natural when omitted: boundary and write algorithms decide. */
  retention?: Retention;
}

export declare class Cortext {
  constructor(
    config?: CortextConfig | null,
    dbPath?: string | null
  );
  constructor(dbPath: string);
  processTextJson(
    text: string,
    sourceId: string,
    options?: ProcessOptions | null
  ): string;
  processText(
    text: string,
    sourceId: string,
    options?: ProcessOptions | null
  ): CortextContext;
  embedTextJson(text: string): string;
  embedText(text: string): number[];
  processAudioJson(
    pcm: Float32Array,
    sourceId: string,
    options?: ProcessOptions | null
  ): string;
  processAudio(
    pcm: Float32Array,
    sourceId: string,
    options?: ProcessOptions | null
  ): CortextContext;
  processAudioWithMediaJson(
    pcm: Float32Array,
    sourceId: string,
    media?: Media | Uint8Array | null,
    mediaMimeType?: string | ProcessOptions | null,
    options?: ProcessOptions | null
  ): string;
  processAudioWithMedia(
    pcm: Float32Array,
    sourceId: string,
    media?: Media | Uint8Array | null,
    mediaMimeType?: string | ProcessOptions | null,
    options?: ProcessOptions | null
  ): CortextContext;
  embedAudioJson(pcm: Float32Array): string;
  embedAudio(pcm: Float32Array): number[];
  processImageJson(
    data: Uint8Array,
    width: number,
    height: number,
    channels: number,
    sourceId: string,
    options?: ProcessOptions | null
  ): string;
  processImage(
    data: Uint8Array,
    width: number,
    height: number,
    channels: number,
    sourceId: string,
    options?: ProcessOptions | null
  ): CortextContext;
  processImageWithMediaJson(
    data: Uint8Array,
    width: number,
    height: number,
    channels: number,
    sourceId: string,
    media?: Media | Uint8Array | null,
    mediaMimeType?: string | ProcessOptions | null,
    options?: ProcessOptions | null
  ): string;
  processImageWithMedia(
    data: Uint8Array,
    width: number,
    height: number,
    channels: number,
    sourceId: string,
    media?: Media | Uint8Array | null,
    mediaMimeType?: string | ProcessOptions | null,
    options?: ProcessOptions | null
  ): CortextContext;
  embedImageJson(
    data: Uint8Array,
    width: number,
    height: number,
    channels: number
  ): string;
  embedImage(
    data: Uint8Array,
    width: number,
    height: number,
    channels: number
  ): number[];
  consolidateJson(): string;
  consolidate(): CortextContext;
  flush(): void;
  reset(): void;
}

export declare function version(): string;
export declare function lastError(): string;
