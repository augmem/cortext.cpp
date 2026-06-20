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

export declare class Cortext {
  constructor(config?: CortextConfig, dbPath?: string, modelsDir?: string);
  processTextJson(text: string, sourceId: string): string;
  processText(text: string, sourceId: string): CortextContext;
  embedTextJson(text: string): string;
  embedText(text: string): number[];
  processAudioJson(pcm: Float32Array, sourceId: string): string;
  processAudio(pcm: Float32Array, sourceId: string): CortextContext;
  embedAudioJson(pcm: Float32Array): string;
  embedAudio(pcm: Float32Array): number[];
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
