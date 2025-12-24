import uvicorn
import httpx
import argparse
import itertools
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response

app = FastAPI(title="MinerU Load Balancer")

# Default backends (can be overridden by args)
BACKEND_URLS = []
iterator = None

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])
async def proxy(request: Request, path_name: str):
    global iterator
    if not iterator:
        return Response("No backends configured", status_code=503)

    # Round-robin selection
    target_base = next(iterator)
    
    # Construct target URL
    url = f"{target_base}/{path_name}"
    if request.url.query:
        url += f"?{request.url.query}"
    
    # Forward headers (excluding Host to avoid conflicts)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None) # Let httpx handle content-length
    
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            # Read body
            body = await request.body()
            
            # Build request
            req = client.build_request(
                request.method,
                url,
                headers=headers,
                content=body,
                timeout=None
            )
            
            # Send request and stream response
            r = await client.send(req, stream=True)
            
            return StreamingResponse(
                r.aiter_raw(),
                status_code=r.status_code,
                headers=r.headers,
                background=None
            )
        except Exception as e:
            # Simple error handling
            import traceback
            traceback.print_exc()
            return Response(content=f"Proxy Error: {str(e)}", status_code=502)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010, help="Port for the load balancer")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the load balancer")
    parser.add_argument("--backends", nargs="+", required=True, help="List of backend URLs (e.g., http://localhost:8011)")
    args = parser.parse_args()
    
    BACKEND_URLS = args.backends
    iterator = itertools.cycle(BACKEND_URLS)
    
    print(f"Starting MinerU Load Balancer on {args.host}:{args.port}")
    print(f"Balancing between backends: {BACKEND_URLS}")
    
    uvicorn.run(app, host=args.host, port=args.port)
