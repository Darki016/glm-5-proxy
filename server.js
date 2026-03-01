// server.js — GLM-5 Reasoning Proxy for SillyTavern & Janitor AI
// Powered by NVIDIA NIM | GLM-5 744B MoE with Thinking Mode

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// ─── CONFIG ───────────────────────────────────────────────────────────────────
const NIM_BASE    = 'https://integrate.api.nvidia.com/v1';
const NIM_KEY     = process.env.NIM_API_KEY;
const GLM5_MODEL  = 'z-ai/glm5';          // exact NVIDIA NIM model ID

// Set true  → show <think>…</think> block before the answer (great for SillyTavern)
// Set false → clean answer only, no reasoning shown (better for Janitor AI)
const SHOW_REASONING = true;
// ──────────────────────────────────────────────────────────────────────────────

// ALL model names from SillyTavern / Janitor AI map to GLM-5
function resolveModel() {
  return GLM5_MODEL;
}

// ─── HEALTH CHECK ─────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    model: GLM5_MODEL,
    reasoning: SHOW_REASONING ? 'visible' : 'hidden',
    context_tokens: 205000
  });
});

// ─── MODELS LIST (OpenAI-compatible) ─────────────────────────────────────────
app.get('/v1/models', (req, res) => {
  // Expose fake model names so SillyTavern / Janitor AI dropdowns populate
  const names = [
    'gpt-4o', 'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo',
    'claude-3-opus', 'claude-3-sonnet', 'gemini-pro', 'glm-5'
  ];
  res.json({
    object: 'list',
    data: names.map(id => ({
      id,
      object: 'model',
      created: 1700000000,
      owned_by: 'glm5-nim-proxy'
    }))
  });
});

// ─── MAIN CHAT ENDPOINT ───────────────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  if (!NIM_KEY) {
    return res.status(500).json({
      error: { message: 'NIM_API_KEY env variable is not set on the server.', type: 'configuration_error' }
    });
  }

  try {
    const { messages = [], temperature, max_tokens, stream } = req.body;
    const useStream = stream === true;

    // Build NIM request — thinking is enabled via chat_template_kwargs
    const nimPayload = {
      model: resolveModel(),
      messages,
      temperature: typeof temperature === 'number' ? temperature : 0.7,
      max_tokens: typeof max_tokens === 'number' ? max_tokens : 8192,
      stream: useStream,
      // ← This is the key that turns on GLM-5 reasoning
      extra_body: {
        chat_template_kwargs: { enable_thinking: true }
      }
    };

    const headers = {
      'Authorization': `Bearer ${NIM_KEY}`,
      'Content-Type': 'application/json'
    };

    // ── STREAMING ─────────────────────────────────────────────────────────────
    if (useStream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders();

      const nimRes = await axios.post(`${NIM_BASE}/chat/completions`, nimPayload, {
        headers,
        responseType: 'stream'
      });

      let buf = '';
      let thinkOpen = false;   // track if <think> tag was opened
      let thinkClosed = false; // track if </think> tag was sent

      nimRes.data.on('data', chunk => {
        buf += chunk.toString();
        const lines = buf.split('\n');
        buf = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data:')) continue;

          const payload = trimmed.slice(5).trim();
          if (payload === '[DONE]') {
            // If reasoning was started but never closed, close it now
            if (SHOW_REASONING && thinkOpen && !thinkClosed) {
              const closeTag = buildDelta('</think>\n\n');
              res.write(`data: ${JSON.stringify(closeTag)}\n\n`);
            }
            res.write('data: [DONE]\n\n');
            continue;
          }

          let parsed;
          try { parsed = JSON.parse(payload); } catch { continue; }

          const delta = parsed?.choices?.[0]?.delta;
          if (!delta) { res.write(`data: ${JSON.stringify(parsed)}\n\n`); continue; }

          const reasoning = delta.reasoning_content || '';
          const content   = delta.content || '';

          if (SHOW_REASONING) {
            // Stream reasoning wrapped in <think> tags, then the answer
            if (reasoning) {
              if (!thinkOpen) {
                // First reasoning chunk — open the tag
                const open = buildDelta('<think>\n' + reasoning);
                res.write(`data: ${JSON.stringify(open)}\n\n`);
                thinkOpen = true;
              } else {
                const rc = buildDelta(reasoning);
                res.write(`data: ${JSON.stringify(rc)}\n\n`);
              }
            }
            if (content) {
              if (thinkOpen && !thinkClosed) {
                // Close reasoning before first answer chunk
                const close = buildDelta('\n</think>\n\n' + content);
                res.write(`data: ${JSON.stringify(close)}\n\n`);
                thinkClosed = true;
              } else {
                const ac = buildDelta(content);
                res.write(`data: ${JSON.stringify(ac)}\n\n`);
              }
            }
          } else {
            // Only forward actual answer content
            if (content) {
              const ac = buildDelta(content);
              res.write(`data: ${JSON.stringify(ac)}\n\n`);
            }
          }
        }
      });

      nimRes.data.on('end', () => res.end());
      nimRes.data.on('error', err => {
        console.error('Stream error:', err.message);
        res.end();
      });

    // ── NON-STREAMING ─────────────────────────────────────────────────────────
    } else {
      const nimRes = await axios.post(`${NIM_BASE}/chat/completions`, nimPayload, { headers });
      const data = nimRes.data;

      const choices = (data.choices || []).map((c, i) => {
        const answer    = c.message?.content || '';
        const reasoning = c.message?.reasoning_content || '';

        let fullContent = answer;
        if (SHOW_REASONING && reasoning) {
          fullContent = `<think>\n${reasoning}\n</think>\n\n${answer}`;
        }

        return {
          index: i,
          message: { role: c.message?.role || 'assistant', content: fullContent },
          finish_reason: c.finish_reason || 'stop'
        };
      });

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: 'glm-5',
        choices,
        usage: data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }

  } catch (err) {
    const status  = err.response?.status || 500;
    const message = err.response?.data?.detail || err.response?.data?.message || err.message || 'Unknown error';
    console.error(`[${status}] NIM Error:`, message);
    res.status(status).json({
      error: { message, type: 'api_error', code: status }
    });
  }
});

// ─── CATCH-ALL ────────────────────────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `${req.path} not found`, type: 'not_found' } });
});

// ─── HELPERS ─────────────────────────────────────────────────────────────────
function buildDelta(text) {
  return {
    id: `chatcmpl-${Date.now()}`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: 'glm-5',
    choices: [{ index: 0, delta: { content: text }, finish_reason: null }]
  };
}

app.listen(PORT, () => {
  console.log(`GLM-5 NIM Proxy running on port ${PORT}`);
  console.log(`Model    : ${GLM5_MODEL}`);
  console.log(`Reasoning: ${SHOW_REASONING ? 'VISIBLE (<think> tags)' : 'HIDDEN (clean output)'}`);
});
