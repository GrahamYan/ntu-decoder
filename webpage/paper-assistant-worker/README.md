# NTU Paper Assistant Worker

Cloudflare Worker backend for the NTU Decoder project page. It receives a user
question, retrieves relevant paper context from local chunks, and calls
Cloudflare Workers AI through the `AI` binding.

## Deploy

```bash
cd paper-assistant-worker
npm install
npx wrangler login
npx wrangler deploy
```

After deploy, Cloudflare prints a URL similar to:

```text
https://ntu-paper-assistant.<your-workers-subdomain>.workers.dev
```

Copy the `/chat` endpoint into the project page:

```html
data-endpoint="https://ntu-paper-assistant.<your-workers-subdomain>.workers.dev/chat"
```

Then commit and push the updated `index.html`.

## Local/remote test

Workers AI runs on Cloudflare, so local development should use remote mode:

```bash
npm run dev
```

Then send a test request:

```bash
curl -X POST "http://127.0.0.1:8787/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"What is Neural Transfer Unification?"}'
```

## Notes

- The frontend never stores model credentials.
- `ALLOWED_ORIGINS` restricts browser calls to the GitHub Pages site and local dev.
- Keep answers grounded in the bundled paper context. If the paper does not
  specify something, the assistant is instructed to say so.
