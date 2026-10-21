import { Hono } from 'hono'

// VE mograph — this Hono app only serves the AE-FX preset GALLERY (static files in public/aefx/).
// The video engine itself is the Python SDK in sdk/ (see README.md / sdk/AEFX_GUIDE.md).
const app = new Hono()

app.get('/', (c) => c.redirect('/aefx/index.html'))
app.get('/aefx', (c) => c.redirect('/aefx/index.html'))
app.get('/aefx/', (c) => c.redirect('/aefx/index.html'))
app.get('/api/health', (c) => c.json({ ok: true, gallery: '/aefx/index.html' }))

export default app
