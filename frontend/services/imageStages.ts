export interface SketchStages {
  stage1: string;
  stage2: string;
  final: string;
  width: number;
  height: number;
}

const loadImage = (src: string): Promise<HTMLImageElement> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.decoding = 'async';
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(err);
    img.src = src;
  });
};

const toGrayscale = (imageData: ImageData): Uint8ClampedArray => {
  const { data, width, height } = imageData;
  const gray = new Uint8ClampedArray(width * height);

  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }

  return gray;
};

const grayscaleToDataUrl = (
  gray: Uint8ClampedArray,
  width: number,
  height: number
): string => {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Canvas 2D context is not available');
  }

  const output = ctx.createImageData(width, height);
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    output.data[i * 4] = v;
    output.data[i * 4 + 1] = v;
    output.data[i * 4 + 2] = v;
    output.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(output, 0, 0);
  return canvas.toDataURL('image/png');
};

const sobelEdges = (
  gray: Uint8ClampedArray,
  width: number,
  height: number
): Uint8ClampedArray => {
  const result = new Uint8ClampedArray(gray.length);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const gx =
        -1 * gray[idx - width - 1] +
        1 * gray[idx - width + 1] +
        -2 * gray[idx - 1] +
        2 * gray[idx + 1] +
        -1 * gray[idx + width - 1] +
        1 * gray[idx + width + 1];

      const gy =
        -1 * gray[idx - width - 1] +
        -2 * gray[idx - width] +
        -1 * gray[idx - width + 1] +
        1 * gray[idx + width - 1] +
        2 * gray[idx + width] +
        1 * gray[idx + width + 1];

      const magnitude = Math.min(255, Math.hypot(gx, gy));
      result[idx] = magnitude;
    }
  }
  return result;
};

const buildPencilLayer = async (
  gray: Uint8ClampedArray,
  width: number,
  height: number,
  grayCanvas: HTMLCanvasElement
): Promise<Uint8ClampedArray> => {
  const blurCanvas = document.createElement('canvas');
  blurCanvas.width = width;
  blurCanvas.height = height;
  const blurCtx = blurCanvas.getContext('2d');
  const grayCtx = grayCanvas.getContext('2d');

  if (!blurCtx || !grayCtx) {
    throw new Error('Canvas 2D context is not available');
  }

  blurCtx.filter = 'blur(16px)';
  blurCtx.drawImage(grayCanvas, 0, 0, width, height);
  const blurred = blurCtx.getImageData(0, 0, width, height).data;

  const pencil = new Uint8ClampedArray(gray.length);
  for (let i = 0; i < gray.length; i++) {
    const inverted = 255 - blurred[i * 4];
    const value = Math.min(255, (gray[i] * 256) / Math.max(1, inverted));
    pencil[i] = value;
  }

  return pencil;
};

export const buildSketchStages = async (imageUrl: string): Promise<SketchStages> => {
  if (typeof document === 'undefined') {
    throw new Error('Image processing is only available in the browser');
  }

  const image = await loadImage(imageUrl);
  const width = image.naturalWidth;
  const height = image.naturalHeight;

  const baseCanvas = document.createElement('canvas');
  baseCanvas.width = width;
  baseCanvas.height = height;
  const baseCtx = baseCanvas.getContext('2d');
  if (!baseCtx) {
    throw new Error('Canvas 2D context is not available');
  }

  baseCtx.drawImage(image, 0, 0, width, height);
  const imageData = baseCtx.getImageData(0, 0, width, height);
  const gray = toGrayscale(imageData);

  const grayCanvas = document.createElement('canvas');
  grayCanvas.width = width;
  grayCanvas.height = height;
  const grayCtx = grayCanvas.getContext('2d');
  if (!grayCtx) {
    throw new Error('Canvas 2D context is not available');
  }

  const grayImageData = grayCtx.createImageData(width, height);
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    grayImageData.data[i * 4] = v;
    grayImageData.data[i * 4 + 1] = v;
    grayImageData.data[i * 4 + 2] = v;
    grayImageData.data[i * 4 + 3] = 255;
  }
  grayCtx.putImageData(grayImageData, 0, 0);

  const edges = sobelEdges(gray, width, height);
  const construction = new Uint8ClampedArray(gray.length);
  for (let i = 0; i < gray.length; i++) {
    const boosted = Math.min(255, edges[i] * 1.2);
    construction[i] = 255 - boosted;
  }

  const pencil = await buildPencilLayer(gray, width, height, grayCanvas);

  return {
    stage1: grayscaleToDataUrl(construction, width, height),
    stage2: grayscaleToDataUrl(pencil, width, height),
    final: baseCanvas.toDataURL('image/png'),
    width,
    height,
  };
};
