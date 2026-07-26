export interface Env {
  TRAINING_DATA_BUCKET: R2Bucket;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    
    // Only allow PUT requests to /upload/:filename
    if (request.method === 'PUT' && url.pathname.startsWith('/upload/')) {
      const filename = url.pathname.split('/').pop();
      if (!filename) return new Response('Bad Request', { status: 400 });

      // Save directly to R2
      await env.TRAINING_DATA_BUCKET.put(filename, request.body);
      
      return new Response('Uploaded successfully', { status: 200 });
    }

    if (request.method === 'GET' && url.pathname === '/list') {
      const list = await env.TRAINING_DATA_BUCKET.list();
      const files = list.objects.map(o => o.key);
      return new Response(JSON.stringify(files), { 
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (request.method === 'GET' && url.pathname.startsWith('/download/')) {
      const filename = url.pathname.split('/').pop();
      if (!filename) return new Response('Bad Request', { status: 400 });

      const object = await env.TRAINING_DATA_BUCKET.get(filename);
      if (!object) return new Response('Not Found', { status: 404 });

      return new Response(object.body, { status: 200 });
    }

    return new Response('Not Found', { status: 404 });
  },
};
