import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegStatic from 'ffmpeg-static';
import { logger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';

// Set ffmpeg path
if (ffmpegStatic) {
  ffmpeg.setFfmpegPath(ffmpegStatic);
}

interface MediaConfig {
  tempDir: string;
  maxFileSize: number;
  supportedImageFormats: string[];
  supportedVideoFormats: string[];
  supportedAudioFormats: string[];
}

interface ProcessingRequest {
  contentType: string;
  contentData: string; // Base64 or file path
  analysisType: string;
  outputFormat?: string;
  quality?: number;
  dimensions?: { width: number; height: number };
}

interface ProcessingResult {
  id: string;
  contentType: string;
  originalSize: number;
  processedSize?: number;
  metadata: Record<string, any>;
  extractedData: any;
  thumbnails?: string[];
  processedAt: string;
  processingTime: number;
}

interface ImageMetadata {
  width: number;
  height: number;
  format: string;
  colorSpace: string;
  channels: number;
  density: number;
  hasAlpha: boolean;
  exif?: Record<string, any>;
}

interface VideoMetadata {
  duration: number;
  bitrate: number;
  fps: number;
  width: number;
  height: number;
  format: string;
  codec: string;
  audioChannels?: number;
  audioCodec?: string;
}

interface AudioMetadata {
  duration: number;
  bitrate: number;
  sampleRate: number;
  channels: number;
  format: string;
  codec: string;
}

export class MediaProcessor {
  private config: MediaConfig;
  private processedFiles: Map<string, ProcessingResult> = new Map();

  constructor(config: Partial<MediaConfig> = {}) {
    this.config = {
      tempDir: config.tempDir || '/tmp/media-processor',
      maxFileSize: config.maxFileSize || 100 * 1024 * 1024, // 100MB
      supportedImageFormats: config.supportedImageFormats || ['jpg', 'jpeg', 'png', 'gif', 'webp', 'tiff', 'bmp'],
      supportedVideoFormats: config.supportedVideoFormats || ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv'],
      supportedAudioFormats: config.supportedAudioFormats || ['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a']
    };

    this.ensureTempDirectory();
    logger.info('Media processor initialized');
  }

  async processMultimodal(request: ProcessingRequest): Promise<ProcessingResult> {
    const startTime = Date.now();
    const processingId = uuidv4();

    try {
      const { contentType, contentData, analysisType } = request;
      
      // Validate input
      this.validateInput(contentType, contentData);
      
      // Prepare file path
      const filePath = await this.prepareInputFile(contentData, contentType);
      const fileStats = fs.statSync(filePath);
      
      if (fileStats.size > this.config.maxFileSize) {
        throw new Error(`File size exceeds maximum limit: ${this.config.maxFileSize} bytes`);
      }

      let result: ProcessingResult;
      
      switch (contentType) {
        case 'image':
          result = await this.processImage(processingId, filePath, analysisType, request);
          break;
        case 'video':
          result = await this.processVideo(processingId, filePath, analysisType, request);
          break;
        case 'audio':
          result = await this.processAudio(processingId, filePath, analysisType, request);
          break;
        case 'document':
          result = await this.processDocument(processingId, filePath, analysisType, request);
          break;
        case 'text':
          result = await this.processText(processingId, contentData, analysisType, request);
          break;
        default:
          throw new Error(`Unsupported content type: ${contentType}`);
      }

      result.originalSize = fileStats.size;
      result.processingTime = Date.now() - startTime;
      result.processedAt = new Date().toISOString();
      
      this.processedFiles.set(processingId, result);
      
      // Clean up temporary file
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
      
      logger.info(`Media processing completed: ${processingId}`);
      return result;
    } catch (error) {
      logger.error(`Media processing failed: ${error.message}`);
      throw error;
    }
  }

  private async processImage(id: string, filePath: string, analysisType: string, request: ProcessingRequest): Promise<ProcessingResult> {
    const image = sharp(filePath);
    const metadata = await image.metadata();
    
    const imageMetadata: ImageMetadata = {
      width: metadata.width || 0,
      height: metadata.height || 0,
      format: metadata.format || 'unknown',
      colorSpace: metadata.space || 'unknown',
      channels: metadata.channels || 0,
      density: metadata.density || 0,
      hasAlpha: metadata.hasAlpha || false,
      exif: metadata.exif ? this.parseExifData(metadata.exif) : undefined
    };

    let extractedData: any = {};
    let thumbnails: string[] = [];

    switch (analysisType) {
      case 'describe':
        extractedData = await this.describeImage(filePath, imageMetadata);
        break;
      case 'analyze':
        extractedData = await this.analyzeImage(filePath, imageMetadata);
        break;
      case 'extract_text':
        extractedData = await this.extractTextFromImage(filePath);
        break;
      case 'classify':
        extractedData = await this.classifyImage(filePath, imageMetadata);
        break;
      default:
        extractedData = { description: 'Basic image processing completed' };
    }

    // Generate thumbnails
    if (request.dimensions || analysisType === 'analyze') {
      thumbnails = await this.generateImageThumbnails(filePath, id);
    }

    // Convert to base64 for response
    const base64Data = fs.readFileSync(filePath).toString('base64');
    extractedData.base64Data = base64Data;

    return {
      id,
      contentType: 'image',
      originalSize: 0, // Will be set by caller
      metadata: imageMetadata,
      extractedData,
      thumbnails,
      processedAt: '',
      processingTime: 0
    };
  }

  private async processVideo(id: string, filePath: string, analysisType: string, request: ProcessingRequest): Promise<ProcessingResult> {
    const videoMetadata = await this.getVideoMetadata(filePath);
    
    let extractedData: any = {};
    let thumbnails: string[] = [];

    switch (analysisType) {
      case 'describe':
        extractedData = await this.describeVideo(filePath, videoMetadata);
        break;
      case 'analyze':
        extractedData = await this.analyzeVideo(filePath, videoMetadata);
        thumbnails = await this.generateVideoThumbnails(filePath, id);
        break;
      case 'extract_text':
        extractedData = await this.extractTextFromVideo(filePath);
        break;
      case 'summarize':
        extractedData = await this.summarizeVideo(filePath, videoMetadata);
        break;
      default:
        extractedData = { description: 'Basic video processing completed' };
    }

    return {
      id,
      contentType: 'video',
      originalSize: 0,
      metadata: videoMetadata,
      extractedData,
      thumbnails,
      processedAt: '',
      processingTime: 0
    };
  }

  private async processAudio(id: string, filePath: string, analysisType: string, request: ProcessingRequest): Promise<ProcessingResult> {
    const audioMetadata = await this.getAudioMetadata(filePath);
    
    let extractedData: any = {};

    switch (analysisType) {
      case 'describe':
        extractedData = await this.describeAudio(filePath, audioMetadata);
        break;
      case 'analyze':
        extractedData = await this.analyzeAudio(filePath, audioMetadata);
        break;
      case 'extract_text':
        extractedData = await this.transcribeAudio(filePath);
        break;
      case 'sentiment':
        extractedData = await this.analyzeAudioSentiment(filePath);
        break;
      default:
        extractedData = { description: 'Basic audio processing completed' };
    }

    return {
      id,
      contentType: 'audio',
      originalSize: 0,
      metadata: audioMetadata,
      extractedData,
      processedAt: '',
      processingTime: 0
    };
  }

  private async processDocument(id: string, filePath: string, analysisType: string, request: ProcessingRequest): Promise<ProcessingResult> {
    const fileExtension = path.extname(filePath).toLowerCase();
    let extractedData: any = {};
    let metadata: any = {};

    try {
      switch (fileExtension) {
        case '.pdf':
          extractedData = await this.processPDF(filePath, analysisType);
          break;
        case '.docx':
          extractedData = await this.processWord(filePath, analysisType);
          break;
        case '.txt':
        case '.md':
          extractedData = await this.processTextFile(filePath, analysisType);
          break;
        default:
          throw new Error(`Unsupported document format: ${fileExtension}`);
      }

      metadata = {
        format: fileExtension,
        size: fs.statSync(filePath).size,
        wordCount: extractedData.text ? extractedData.text.split(/\s+/).length : 0,
        pageCount: extractedData.pageCount || 1
      };
    } catch (error) {
      logger.error(`Document processing error: ${error.message}`);
      extractedData = { error: error.message };
    }

    return {
      id,
      contentType: 'document',
      originalSize: 0,
      metadata,
      extractedData,
      processedAt: '',
      processingTime: 0
    };
  }

  private async processText(id: string, textData: string, analysisType: string, request: ProcessingRequest): Promise<ProcessingResult> {
    // Decode base64 if needed
    let text = textData;
    try {
      if (this.isBase64(textData)) {
        text = Buffer.from(textData, 'base64').toString('utf-8');
      }
    } catch (error) {
      // If decoding fails, treat as plain text
    }

    let extractedData: any = {};
    
    switch (analysisType) {
      case 'analyze':
        extractedData = await this.analyzeText(text);
        break;
      case 'sentiment':
        extractedData = await this.analyzeTextSentiment(text);
        break;
      case 'summarize':
        extractedData = await this.summarizeText(text);
        break;
      case 'classify':
        extractedData = await this.classifyText(text);
        break;
      default:
        extractedData = { text, length: text.length };
    }

    const metadata = {
      length: text.length,
      wordCount: text.split(/\s+/).length,
      lineCount: text.split('\n').length,
      language: await this.detectLanguage(text)
    };

    return {
      id,
      contentType: 'text',
      originalSize: Buffer.byteLength(text, 'utf8'),
      metadata,
      extractedData,
      processedAt: '',
      processingTime: 0
    };
  }

  // Helper methods for different media types
  private async describeImage(filePath: string, metadata: ImageMetadata): Promise<any> {
    return {
      description: `Image with dimensions ${metadata.width}x${metadata.height} in ${metadata.format} format`,
      colorSpace: metadata.colorSpace,
      channels: metadata.channels,
      hasAlpha: metadata.hasAlpha,
      analysis: 'Image description requires AI model integration'
    };
  }

  private async analyzeImage(filePath: string, metadata: ImageMetadata): Promise<any> {
    // Analyze image properties
    const image = sharp(filePath);
    const stats = await image.stats();
    
    return {
      dimensions: { width: metadata.width, height: metadata.height },
      format: metadata.format,
      colorAnalysis: {
        dominantColors: this.extractDominantColors(stats),
        brightness: this.calculateBrightness(stats),
        contrast: this.calculateContrast(stats)
      },
      technicalDetails: {
        fileSize: fs.statSync(filePath).size,
        colorSpace: metadata.colorSpace,
        channels: metadata.channels,
        density: metadata.density
      }
    };
  }

  private async extractTextFromImage(filePath: string): Promise<any> {
    // OCR functionality would go here
    return {
      text: 'OCR text extraction requires integration with Tesseract or similar OCR library',
      confidence: 0.0,
      language: 'unknown'
    };
  }

  private async classifyImage(filePath: string, metadata: ImageMetadata): Promise<any> {
    // Image classification logic
    return {
      categories: ['object', 'scene', 'concept'],
      predictions: [
        { label: 'Unknown', confidence: 0.0 }
      ],
      note: 'Image classification requires AI model integration'
    };
  }

  private async generateImageThumbnails(filePath: string, id: string): Promise<string[]> {
    const thumbnails: string[] = [];
    const sizes = [{ width: 150, height: 150 }, { width: 300, height: 300 }];
    
    for (const size of sizes) {
      const thumbnailPath = path.join(this.config.tempDir, `${id}_${size.width}x${size.height}.jpg`);
      
      await sharp(filePath)
        .resize(size.width, size.height, {
          fit: 'cover',
          position: 'center'
        })
        .jpeg({ quality: 80 })
        .toFile(thumbnailPath);
      
      const base64 = fs.readFileSync(thumbnailPath).toString('base64');
      thumbnails.push(`data:image/jpeg;base64,${base64}`);
      
      // Clean up thumbnail file
      fs.unlinkSync(thumbnailPath);
    }
    
    return thumbnails;
  }

  private async getVideoMetadata(filePath: string): Promise<VideoMetadata> {
    return new Promise((resolve, reject) => {
      ffmpeg.ffprobe(filePath, (err, metadata) => {
        if (err) {
          reject(err);
          return;
        }

        const videoStream = metadata.streams.find(stream => stream.codec_type === 'video');
        const audioStream = metadata.streams.find(stream => stream.codec_type === 'audio');

        if (!videoStream) {
          reject(new Error('No video stream found'));
          return;
        }

        resolve({
          duration: parseFloat(metadata.format.duration || '0'),
          bitrate: parseInt(metadata.format.bit_rate || '0'),
          fps: this.parseFPS(videoStream.r_frame_rate || '0/1'),
          width: videoStream.width || 0,
          height: videoStream.height || 0,
          format: metadata.format.format_name || 'unknown',
          codec: videoStream.codec_name || 'unknown',
          audioChannels: audioStream?.channels,
          audioCodec: audioStream?.codec_name
        });
      });
    });
  }

  private async generateVideoThumbnails(filePath: string, id: string): Promise<string[]> {
    const thumbnails: string[] = [];
    const timestamps = ['00:00:01', '00:00:05', '00:00:10'];
    
    for (let i = 0; i < timestamps.length; i++) {
      const thumbnailPath = path.join(this.config.tempDir, `${id}_thumb_${i}.jpg`);
      
      await new Promise<void>((resolve, reject) => {
        ffmpeg(filePath)
          .seekInput(timestamps[i])
          .frames(1)
          .size('300x200')
          .output(thumbnailPath)
          .on('end', () => resolve())
          .on('error', reject)
          .run();
      });
      
      if (fs.existsSync(thumbnailPath)) {
        const base64 = fs.readFileSync(thumbnailPath).toString('base64');
        thumbnails.push(`data:image/jpeg;base64,${base64}`);
        fs.unlinkSync(thumbnailPath);
      }
    }
    
    return thumbnails;
  }

  private async getAudioMetadata(filePath: string): Promise<AudioMetadata> {
    return new Promise((resolve, reject) => {
      ffmpeg.ffprobe(filePath, (err, metadata) => {
        if (err) {
          reject(err);
          return;
        }

        const audioStream = metadata.streams.find(stream => stream.codec_type === 'audio');
        if (!audioStream) {
          reject(new Error('No audio stream found'));
          return;
        }

        resolve({
          duration: parseFloat(metadata.format.duration || '0'),
          bitrate: parseInt(metadata.format.bit_rate || '0'),
          sampleRate: audioStream.sample_rate || 0,
          channels: audioStream.channels || 0,
          format: metadata.format.format_name || 'unknown',
          codec: audioStream.codec_name || 'unknown'
        });
      });
    });
  }

  private async processPDF(filePath: string, analysisType: string): Promise<any> {
    // PDF processing would require pdf-parse or similar
    return {
      text: 'PDF processing requires pdf-parse library integration',
      pageCount: 1,
      analysis: 'PDF analysis placeholder'
    };
  }

  private async processWord(filePath: string, analysisType: string): Promise<any> {
    // Word processing would require mammoth or similar
    return {
      text: 'Word document processing requires mammoth library integration',
      analysis: 'Word analysis placeholder'
    };
  }

  private async processTextFile(filePath: string, analysisType: string): Promise<any> {
    const text = fs.readFileSync(filePath, 'utf-8');
    return {
      text,
      lineCount: text.split('\n').length,
      wordCount: text.split(/\s+/).length,
      analysis: await this.analyzeText(text)
    };
  }

  // Text analysis methods
  private async analyzeText(text: string): Promise<any> {
    return {
      length: text.length,
      wordCount: text.split(/\s+/).length,
      sentenceCount: text.split(/[.!?]+/).length,
      paragraphCount: text.split(/\n\s*\n/).length,
      readingTime: Math.ceil(text.split(/\s+/).length / 200), // Assuming 200 WPM
      language: await this.detectLanguage(text)
    };
  }

  private async analyzeTextSentiment(text: string): Promise<any> {
    // Sentiment analysis placeholder
    return {
      sentiment: 'neutral',
      confidence: 0.5,
      emotions: {
        joy: 0.2,
        sadness: 0.1,
        anger: 0.1,
        fear: 0.1,
        surprise: 0.1
      }
    };
  }

  private async summarizeText(text: string): Promise<any> {
    // Text summarization placeholder
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    return {
      summary: sentences.slice(0, 3).join('. ') + '.',
      keyPoints: sentences.slice(0, 5),
      originalLength: text.length,
      summaryLength: sentences.slice(0, 3).join('. ').length
    };
  }

  private async classifyText(text: string): Promise<any> {
    // Text classification placeholder
    return {
      category: 'general',
      subcategories: ['text', 'content'],
      confidence: 0.7,
      topics: ['general discussion']
    };
  }

  private async detectLanguage(text: string): Promise<string> {
    // Simple language detection placeholder
    const commonEnglishWords = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'as', 'with'];
    const words = text.toLowerCase().split(/\s+/);
    const englishScore = words.filter(word => commonEnglishWords.includes(word)).length;
    return englishScore > words.length * 0.1 ? 'english' : 'unknown';
  }

  // Video analysis methods
  private async describeVideo(filePath: string, metadata: VideoMetadata): Promise<any> {
    return {
      description: `Video with duration ${metadata.duration.toFixed(2)}s, ${metadata.width}x${metadata.height} resolution`,
      duration: metadata.duration,
      fps: metadata.fps,
      resolution: `${metadata.width}x${metadata.height}`,
      codec: metadata.codec,
      hasAudio: !!metadata.audioCodec
    };
  }

  private async analyzeVideo(filePath: string, metadata: VideoMetadata): Promise<any> {
    return {
      technicalDetails: metadata,
      quality: this.assessVideoQuality(metadata),
      estimatedFileSize: fs.statSync(filePath).size,
      analysis: 'Video content analysis requires AI model integration'
    };
  }

  private async extractTextFromVideo(filePath: string): Promise<any> {
    return {
      text: 'Video text extraction requires OCR and speech-to-text integration',
      method: 'subtitle_extraction + speech_recognition',
      confidence: 0.0
    };
  }

  private async summarizeVideo(filePath: string, metadata: VideoMetadata): Promise<any> {
    return {
      summary: 'Video summarization requires AI model integration',
      keyframes: 'Keyframe extraction not implemented',
      duration: metadata.duration,
      highlights: []
    };
  }

  // Audio analysis methods
  private async describeAudio(filePath: string, metadata: AudioMetadata): Promise<any> {
    return {
      description: `Audio with duration ${metadata.duration.toFixed(2)}s, ${metadata.sampleRate}Hz sample rate`,
      duration: metadata.duration,
      channels: metadata.channels,
      sampleRate: metadata.sampleRate,
      codec: metadata.codec,
      bitrate: metadata.bitrate
    };
  }

  private async analyzeAudio(filePath: string, metadata: AudioMetadata): Promise<any> {
    return {
      technicalDetails: metadata,
      quality: this.assessAudioQuality(metadata),
      estimatedFileSize: fs.statSync(filePath).size,
      analysis: 'Audio content analysis requires AI model integration'
    };
  }

  private async transcribeAudio(filePath: string): Promise<any> {
    return {
      text: 'Audio transcription requires speech-to-text integration',
      confidence: 0.0,
      language: 'unknown',
      speakers: 1
    };
  }

  private async analyzeAudioSentiment(filePath: string): Promise<any> {
    return {
      sentiment: 'neutral',
      confidence: 0.5,
      emotions: {
        joy: 0.2,
        sadness: 0.1,
        anger: 0.1,
        fear: 0.1,
        surprise: 0.1
      },
      note: 'Audio sentiment analysis requires speech-to-text and sentiment analysis integration'
    };
  }

  // Utility methods
  private validateInput(contentType: string, contentData: string): void {
    if (!contentType || !contentData) {
      throw new Error('Content type and data are required');
    }

    const supportedTypes = ['image', 'video', 'audio', 'document', 'text'];
    if (!supportedTypes.includes(contentType)) {
      throw new Error(`Unsupported content type: ${contentType}`);
    }
  }

  private async prepareInputFile(contentData: string, contentType: string): Promise<string> {
    if (this.isBase64(contentData)) {
      // Handle base64 data
      const buffer = Buffer.from(contentData, 'base64');
      const extension = this.getExtensionFromContentType(contentType);
      const fileName = `${uuidv4()}.${extension}`;
      const filePath = path.join(this.config.tempDir, fileName);
      
      fs.writeFileSync(filePath, buffer);
      return filePath;
    } else if (this.isFilePath(contentData)) {
      // Handle file path
      if (!fs.existsSync(contentData)) {
        throw new Error(`File not found: ${contentData}`);
      }
      return contentData;
    } else {
      // Handle direct text content
      const fileName = `${uuidv4()}.txt`;
      const filePath = path.join(this.config.tempDir, fileName);
      fs.writeFileSync(filePath, contentData, 'utf-8');
      return filePath;
    }
  }

  private isBase64(str: string): boolean {
    try {
      return Buffer.from(str, 'base64').toString('base64') === str;
    } catch (err) {
      return false;
    }
  }

  private isFilePath(str: string): boolean {
    return str.includes('/') || str.includes('\\') || str.includes('.');
  }

  private getExtensionFromContentType(contentType: string): string {
    switch (contentType) {
      case 'image': return 'jpg';
      case 'video': return 'mp4';
      case 'audio': return 'mp3';
      case 'document': return 'pdf';
      case 'text': return 'txt';
      default: return 'bin';
    }
  }

  private ensureTempDirectory(): void {
    if (!fs.existsSync(this.config.tempDir)) {
      fs.mkdirSync(this.config.tempDir, { recursive: true });
    }
  }

  private parseExifData(exifBuffer: Buffer): Record<string, any> {
    // EXIF parsing placeholder
    return {
      note: 'EXIF parsing requires exif-parser or similar library'
    };
  }

  private extractDominantColors(stats: any): string[] {
    // Color extraction placeholder
    return ['#FFFFFF', '#000000', '#808080'];
  }

  private calculateBrightness(stats: any): number {
    // Brightness calculation placeholder
    return 0.5;
  }

  private calculateContrast(stats: any): number {
    // Contrast calculation placeholder
    return 0.5;
  }

  private parseFPS(frameRate: string): number {
    const [numerator, denominator] = frameRate.split('/').map(Number);
    return denominator ? numerator / denominator : 0;
  }

  private assessVideoQuality(metadata: VideoMetadata): string {
    const resolution = metadata.width * metadata.height;
    if (resolution >= 1920 * 1080) return 'high';
    if (resolution >= 1280 * 720) return 'medium';
    return 'low';
  }

  private assessAudioQuality(metadata: AudioMetadata): string {
    if (metadata.sampleRate >= 48000 && metadata.bitrate >= 128000) return 'high';
    if (metadata.sampleRate >= 44100 && metadata.bitrate >= 64000) return 'medium';
    return 'low';
  }

  async getProcessingResult(id: string): Promise<ProcessingResult | null> {
    return this.processedFiles.get(id) || null;
  }

  async listProcessedFiles(): Promise<ProcessingResult[]> {
    return Array.from(this.processedFiles.values());
  }

  async cleanup(): Promise<void> {
    try {
      // Clean up temporary files
      if (fs.existsSync(this.config.tempDir)) {
        const files = fs.readdirSync(this.config.tempDir);
        for (const file of files) {
          const filePath = path.join(this.config.tempDir, file);
          if (fs.statSync(filePath).isFile()) {
            fs.unlinkSync(filePath);
          }
        }
      }
      
      this.processedFiles.clear();
      logger.info('Media processor cleanup completed');
    } catch (error) {
      logger.error('Error during media processor cleanup:', error);
    }
  }
}
