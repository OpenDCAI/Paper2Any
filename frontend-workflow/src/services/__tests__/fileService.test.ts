import { beforeEach, describe, expect, it, vi } from 'vitest';

const { backendFetch } = vi.hoisted(() => ({
  backendFetch: vi.fn(),
}));

vi.mock('../../lib/supabase', () => ({
  isSupabaseConfigured: () => false,
  supabase: {
    auth: {
      getSession: vi.fn(),
    },
  },
}));

vi.mock('../backendClient', () => ({
  backendFetch,
}));

import { getFileRecords, uploadAndSaveFile } from '../fileService';

describe('fileService local fallback', () => {
  beforeEach(() => {
    backendFetch.mockReset();
  });

  it('uploads files when Supabase is not configured', async () => {
    backendFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          success: true,
          file_name: 'paper2ppt_html_editable.pptx',
          file_size: 10,
          workflow_type: 'paper2ppt',
          file_path: '/outputs/system/paper2ppt/123/paper2ppt_html_editable.pptx',
          created_at: '2026-05-11T00:00:00',
        }),
        { status: 200 },
      ),
    );

    const record = await uploadAndSaveFile(
      new Blob(['pptx-bytes']),
      'paper2ppt_html_editable.pptx',
      'paper2ppt',
    );

    expect(record?.download_url).toBe('/outputs/system/paper2ppt/123/paper2ppt_html_editable.pptx');
    expect(backendFetch).toHaveBeenCalledWith(
      '/api/v1/files/upload',
      expect.objectContaining({
        method: 'POST',
        body: expect.any(FormData),
      }),
    );
  });

  it('loads history when Supabase is not configured', async () => {
    backendFetch.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          success: true,
          files: [
            {
              file_name: 'paper2ppt_html_editable.pptx',
              workflow_type: 'paper2ppt',
              download_url: '/outputs/system/paper2ppt/123/paper2ppt_html_editable.pptx',
            },
          ],
        }),
        { status: 200 },
      ),
    );

    const files = await getFileRecords();

    expect(files).toHaveLength(1);
    expect(files[0].file_name).toBe('paper2ppt_html_editable.pptx');
    expect(backendFetch).toHaveBeenCalledWith('/api/v1/files/history');
  });
});
