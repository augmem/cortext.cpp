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

export type CortextContext = Record<string, unknown>;
export interface CortextEmbedding {
  embedding: number[];
  dimension: number;
}

export interface Media {
  data: Uint8Array;
  mimetype: string;
}

export interface ProcessOptions {
  includeEmbedding?: boolean;
  omitEmbedding?: boolean;
}

export declare class Cortext {
  constructor(config?: CortextConfig, dbPath?: string, modelsDir?: string);
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
