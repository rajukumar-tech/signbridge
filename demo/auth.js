// Shared auth helpers for the SignBridge demo pages. Talks to the FastAPI backend in ../backend.
export const API_BASE = "http://localhost:8000";
const TOKEN_KEY = "signbridge.token";
const USER_KEY = "signbridge.user";

function store(key, value) {
  try { value == null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch {}
}
function read(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

export function currentUser() {
  try { return JSON.parse(read(USER_KEY)); } catch { return null; }
}
export function token() { return read(TOKEN_KEY); }

export function logout() { store(TOKEN_KEY, null); store(USER_KEY, null); }

function saveSession({ access_token, user }) {
  store(TOKEN_KEY, access_token);
  store(USER_KEY, JSON.stringify(user));
  return user;
}

// Turn a FastAPI error body into one readable sentence.
function errorMessage(body, status) {
  const d = body?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d) && d.length) {
    return d.map((e) => {
      const field = e.loc?.at(-1);
      if (field === "password" && e.type === "string_too_short") return "Password must be at least 8 characters.";
      if (field === "email") return "Please enter a valid email address.";
      return e.msg;
    }).join(" ");
  }
  return `Something went wrong (HTTP ${status}).`;
}

async function post(path, payload) {
  let res;
  try {
    res = await fetch(API_BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(`Can't reach the SignBridge server at ${API_BASE}. Is the backend running?`);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(errorMessage(body, res.status));
  return body;
}

export async function login(email, password) {
  return saveSession(await post("/auth/login", { email, password }));
}

export async function signup({ email, password, display_name, role }) {
  return saveSession(await post("/auth/register", { email, password, display_name: display_name || null, role }));
}

// Fills an element with "Log in / Sign up" links, or the user's name and a Log out button.
export function renderAccount(el) {
  const user = currentUser();
  if (user) {
    el.innerHTML = "";
    const name = document.createElement("span");
    name.textContent = user.display_name || user.email;
    const out = document.createElement("button");
    out.textContent = "Log out";
    out.onclick = () => { logout(); renderAccount(el); };
    el.append(name, out);
  } else {
    el.innerHTML = '<a href="login.html">Log in</a><a href="signup.html" class="primary-link">Sign up</a>';
  }
}
