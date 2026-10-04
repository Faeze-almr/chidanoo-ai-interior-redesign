const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8000/api'

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'

export interface GenerateRoomResponse {
  job_id: string
  status: JobStatus
}

export interface JobStatusResponse {
  job_id: string
  status: JobStatus
  error: string | null
  result_url: string | null
  created_at: number
  started_at: number | null
  finished_at: number | null
}

export interface GenerateRoomParams {
  image: File
  styleKey?: string
  userText?: string
  depthScale?: number
  cannyScale?: number
  numInferenceSteps?: number
  guidanceScale?: number
  seed?: number
}

export async function submitRoomRedesign(params: GenerateRoomParams): Promise<GenerateRoomResponse> {
  const form = new FormData()
  form.append('image', params.image)
  if (params.styleKey) form.append('style_key', params.styleKey)
  if (params.userText) form.append('user_text', params.userText)
  if (params.depthScale !== undefined) form.append('depth_scale', String(params.depthScale))
  if (params.cannyScale !== undefined) form.append('canny_scale', String(params.cannyScale))
  if (params.numInferenceSteps !== undefined) form.append('num_inference_steps', String(params.numInferenceSteps))
  if (params.guidanceScale !== undefined) form.append('guidance_scale', String(params.guidanceScale))
  if (params.seed !== undefined) form.append('seed', String(params.seed))

  const response = await fetch(`${API_BASE_URL}/generate/room`, {
    method: 'POST',
    body: form,
  })
  if (!response.ok) {
    throw new Error((await safeReadDetail(response)) ?? `Request failed with status ${response.status}`)
  }
  return (await response.json()) as GenerateRoomResponse
}

export interface GenerateRegionParams extends GenerateRoomParams {
  mask: File;
  targetDescription: string;
  strength?: number;
}

export async function submitRegionEdit(params: GenerateRegionParams): Promise<GenerateRoomResponse> {
  const form = new FormData()
  form.append('image', params.image)
  form.append('mask_image', params.mask)
  form.append('target_description', params.targetDescription)
  if (params.styleKey) form.append('style_key', params.styleKey)
  if (params.userText) form.append('user_text', params.userText)
  if (params.depthScale !== undefined) form.append('depth_scale', String(params.depthScale))
  if (params.cannyScale !== undefined) form.append('canny_scale', String(params.cannyScale))
  if (params.numInferenceSteps !== undefined) form.append('num_inference_steps', String(params.numInferenceSteps))
  if (params.guidanceScale !== undefined) form.append('guidance_scale', String(params.guidanceScale))
  if (params.seed !== undefined) form.append('seed', String(params.seed))
  if (params.strength !== undefined) form.append('strength', String(params.strength))

  const response = await fetch(`${API_BASE_URL}/edit/region`, {
    method: 'POST',
    body: form,
  })
  if (!response.ok) {
    throw new Error((await safeReadDetail(response)) ?? `Request failed with status ${response.status}`)
  }
  return (await response.json()) as GenerateRoomResponse
}

export interface GenerateReferenceParams extends GenerateRoomParams {
  reference: File;
  referenceScale?: number;
}

export async function submitReferenceRedesign(params: GenerateReferenceParams): Promise<GenerateRoomResponse> {
  const form = new FormData()
  form.append('image', params.image)
  form.append('reference_image', params.reference)
  if (params.styleKey) form.append('style_key', params.styleKey)
  if (params.userText) form.append('user_text', params.userText)
  if (params.depthScale !== undefined) form.append('depth_scale', String(params.depthScale))
  if (params.cannyScale !== undefined) form.append('canny_scale', String(params.cannyScale))
  if (params.referenceScale !== undefined) form.append('reference_scale', String(params.referenceScale))
  if (params.numInferenceSteps !== undefined) form.append('num_inference_steps', String(params.numInferenceSteps))
  if (params.guidanceScale !== undefined) form.append('guidance_scale', String(params.guidanceScale))
  if (params.seed !== undefined) form.append('seed', String(params.seed))

  const response = await fetch(`${API_BASE_URL}/reference/room`, {
    method: 'POST',
    body: form,
  })
  if (!response.ok) {
    throw new Error((await safeReadDetail(response)) ?? `Request failed with status ${response.status}`)
  }
  return (await response.json()) as GenerateRoomResponse
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`)
  if (!response.ok) {
    throw new Error((await safeReadDetail(response)) ?? `Request failed with status ${response.status}`)
  }
  return (await response.json()) as JobStatusResponse
}

export function jobImageUrl(jobId: string, download = false): string {
  return `${API_BASE_URL}/jobs/${jobId}/image${download ? '?download=true' : ''}`
}

export interface HistoryEntry {
  id: string;
  created_at: string;
  mode: 'full_redesign' | 'targeted_inpaint' | 'reference_redesign';
  style: string | null;
  user_request: string | null;
  generation: {
    runtime_sec?: number;
    seed?: number;
  };
}

export async function listHistory(limit = 100): Promise<HistoryEntry[]> {
  const response = await fetch(`${API_BASE_URL}/history?limit=${limit}`)
  if (!response.ok) {
    throw new Error((await safeReadDetail(response)) ?? `Request failed with status ${response.status}`)
  }
  return (await response.json()) as HistoryEntry[]
}

export function historyImageUrl(
  entryId: string,
  asset: 'input' | 'output' | 'reference' | 'mask' = 'output',
  download = false,
): string {
  return `${API_BASE_URL}/history/${entryId}/${asset}${download ? '?download=true' : ''}`
}

export function historyExportUrl(): string {
  return `${API_BASE_URL}/history/export`
}

async function safeReadDetail(response: Response): Promise<string | null> {
  try {
    const data: unknown = await response.json()
    if (data && typeof data === 'object' && 'detail' in data && typeof (data as { detail: unknown }).detail === 'string') {
      return (data as { detail: string }).detail
    }
    return null
  } catch {
    return null
  }
}
