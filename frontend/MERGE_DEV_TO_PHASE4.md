# Merge: dev → feature/issue#277/phase4_lamba_port
> **Date:** 2026-05-05  
> **Branch:** `feature/issue#277/phase4_lamba_port`  
> **Source:** `dev`
---
## Summary
Merge of `dev` into the Phase 4 (file evaluation) branch. This brought in resilience fixes (#352, #353), new services, and bug fixes from the main development line.
---
## Key Decisions
### 1. `svelte-app/` directory removed
The entire `frontend/svelte-app/` directory was deleted during the monorepo migration to `creator-app`. Dev still had this directory with updated files (`authService.js`, `package.json`, `config.js.sample`), but these were **deleted by us** because the equivalent functionality now lives in:
| Old (svelte-app) | New (creator-app / @lamb/ui) |
|------------------|------------------------------|
| `svelte-app/package.json` | `creator-app/package.json` |
| `svelte-app/src/lib/services/authService.js` | `ui/src/lib/services/authService.js` |
| `svelte-app/static/config.js.sample` | `creator-app/src/lib/config.js` + `docker-entrypoint.py` |
### 2. `sanitize.js` placed in `@lamb/ui`
Dev had `sanitize.js` in `creator-app/src/lib/utils/`. We moved it to `ui/src/lib/utils/` because it's a pure utility (marked + DOMPurify) that any module can reuse without app-specific dependencies.
### 3. `apiClient.js` placed in `creator-app` (not `@lamb/ui`)
Dev's `apiClient.js` depends on `$lib/config` (`getApiUrl`) and `$lib/session/sessionManager`, which are app-specific. It was placed in `creator-app/src/lib/services/` to avoid creating a circular dependency. This means **module-chat and module-file-eval cannot use it yet** — they still use raw `fetch()`.
### 4. `authService` kept in `@lamb/ui`
The `authService` object pattern (exporting a single object with all auth methods) was kept from HEAD. Dev used individual function exports. The object pattern is already used by `Login.svelte`, `Signup.svelte`, and `userStore.js`, so changing it would require updating all callers.
### 5. `Nav.svelte` imports kept as relative paths
Dev used `$lib/` imports for `Nav.svelte`, but since it lives inside `@lamb/ui`, the relative paths (`../i18n`, `./LanguageSelector`, `../version`) are correct. Added `getConfig` import that dev had introduced.
### 6. `config.js.sample` deleted
This was a reference template for `window.LAMB_CONFIG`. The actual config is generated dynamically by `backend/docker-entrypoint.py` at container start, and the same information is documented in CLAUDE.md and the entrypoint source. No longer needed as a standalone file.
---
## What from dev did NOT make it (and why)
| Feature | File | Reason |
|---------|------|--------|
| `apiFetch` in `authService.fetchUserProfile()` | `ui/src/lib/services/authService.js` | Circular dependency: `apiClient` is in `creator-app`, `authService` is in `@lamb/ui`. `fetchUserProfile` still uses raw `fetch()` without 401 global handling. |
| `apiClient` in `@lamb/ui` | N/A | Would require decoupling from `$lib/config` and `$lib/session/sessionManager`. Deferred to future refactor. |
| Module-chat/file-eval using `apiFetch` | N/A | These modules don't have `$lib/config.js` and use raw `fetch()` with relative paths. Would need `apiClient` to be generic first. |
| Two `apiClient.js` files coexisting ~~[SOLVED → PS-1]~~ | `utils/apiClient.js` + `services/apiClient.js` | Both exist in creator-app with overlapping responsibilities. `utils/` protects against disabled accounts (`sessionGuard`), `services/` protects against expired tokens (401 → redirect). Not merged yet. |
---
## TODOs for future work
### High Priority
1. **Merge the two `apiClient.js` files into one** ~~[SOLVED → PS-1]~~
   - `utils/apiClient.js` → `authenticatedFetch` + sessionGuard (disabled account detection)
   - `services/apiClient.js` → `apiFetch` + `apiJson` + `apiAxios` (401 global handling)
   - Create a single unified client that has **both** protections
   - Update all services to use the unified client
   - Delete the duplicate
2. **Move `apiClient.js` to `@lamb/ui`**
   - Replace `getApiUrl()` dependency → read `window.LAMB_CONFIG.api.baseUrl` directly
   - Replace `$lib/session/sessionManager` dependency → make it injectable or use dynamic import
   - This enables **all modules** (creator-app, module-chat, module-file-eval) to use `apiFetch` with global 401 handling
   - Eliminates the circular dependency blocking `authService` improvements
3. **Update `authService.fetchUserProfile()` to use `apiFetch`**
   - Currently uses raw `fetch('/creator/me')` — no 401 global handling
   - Once `apiClient` is in `@lamb/ui`, change to `apiFetch('/me', { token })`
   - This matches what dev's version of `authService` already did
### Medium Priority
4. **Migrate remaining `fetch()` raw calls in creator-app to `apiFetch`**
   - `assistantService.js` — ~10 calls using raw `fetch()`
   - `+page.svelte` (org-admin) — uses `axios` directly instead of `apiAxios`
   - Any other services still using raw `fetch()` or direct `axios` imports
5. **Migrate module-chat and module-file-eval to `apiFetch`**
   - module-chat: 7 calls with raw `fetch()` using `?token=` query params
   - module-file-eval: 4 calls with raw `fetch()` with manual auth headers
   - Requires `apiClient` to be in `@lamb/ui` first (see #2)
### Low Priority

6. **Document `window.LAMB_CONFIG` structure**
   - `config.js.sample` was deleted during this merge
   - Consider adding a proper documentation page or keeping a reference in `docs/`

7. **Move `getConfig()` to `@lamb/ui` (fix hardcoded workaround in Nav.svelte)**
   - **Current workaround:** `Nav.svelte` has a hardcoded `const getConfig = () => window.LAMB_CONFIG || { features: {} };` because `config.js` only exists in `creator-app` and `@lamb/ui` cannot import from it (circular dependency).
   - **Proper fix:** Create `@lamb/ui/src/lib/config.js` with `getConfig()` (just reads `window.LAMB_CONFIG`). Keep `getApiUrl()` and `getLambApiUrl()` in `creator-app/src/lib/config.js` since those are app-specific. Update `Nav.svelte` to import `getConfig` from `@lamb/ui`.

8. **Migrate 4 files from `marked` to `renderMarkdownSafe` from `@lamb/ui`**
   - **Current state:** 4 files in `creator-app` import `marked` directly and use `{@html marked(...)}` without sanitization (XSS risk):
     - `routes/+page.svelte` (line 9)
     - `lib/components/ChatInterface.svelte` (line 4)
     - `routes/agent/history/[id]/+page.svelte` (line 8)
     - `lib/components/aac/AacTerminal.svelte` (line 5)
   - **Proper fix:** Replace `import { marked } from 'marked'` + `{@html marked(text)}` with `import { renderMarkdownSafe } from '@lamb/ui'` + `{@html renderMarkdownSafe(text)}`. This eliminates the need for `marked` as a direct dependency in `creator-app` and ensures all markdown rendering is sanitized.

9. **Fix `@lamb/ui/i18n` subpath imports in Pagination.svelte and RubricMetadataForm.svelte**
   - **Risk:** Compilation failure if Vite/SvelteKit doesn't resolve subpath exports correctly
   - **Files:** `lib/components/common/Pagination.svelte`, `lib/components/evaluaitor/RubricMetadataForm.svelte`
   - **Current:** `import { _, locale } from '@lamb/ui/i18n';`
   - **Fix:** Change to `import { _, locale } from '@lamb/ui';` for consistency and reliability

10. **Remove dead imports in `+page.svelte` (root)**
    - **Risk:** None (dead code), but adds confusion and unnecessary dependencies
    - **Files:** `routes/+page.svelte`
    - **Remove:** `import { marked } from 'marked';` (line 9, unused — replaced by `renderMarkdownSafe`)
    - **Remove:** `import { getApiUrl } from '$lib/config';` (line 10, unused — `apiFetch` resolves URLs internally)

11. **Add sanitization to `module-file-eval` markdown rendering**
    - **Risk:** XSS if AI-generated content contains malicious payloads
    - **File:** `module-file-eval/src/routes/grading/+page.svelte`
    - **Current:** Uses `{@html marked.parse(md)}` without DOMPurify sanitization
    - **Fix:** Import and use `renderMarkdownSafe` from `@lamb/ui` (once available as a shared export)

12. **Migrate `createEventDispatcher` to Svelte 5 callback props (~15 components)**
    - **Risk:** Low — Svelte 5 has backwards compat, but deprecated and generates warnings
    - **Affected:** `Login.svelte`, `Signup.svelte`, `Pagination.svelte`, and ~12 more components
    - **Fix:** Replace `dispatch('event-name')` + `on:event-name` with callback props (`oneventname={...}`)

13. **Review `hooks.server.js` i18n `locale.set()` in SSR context**
    - **Risk:** Low — `svelte-i18n` handles this internally, but fragile if `setupI18n()` hasn't run yet
    - **File:** `creator-app/src/hooks.server.js`
    - **Current:** Imports `locale` from `@lamb/ui` and calls `locale.set(lang)` in server hooks
    - **Fix:** If issues arise, move i18n setup logic to `+layout.js` where it's guaranteed to run after `setupI18n()`
---
## Problems Solved

### PS-1 — Consolidate API clients and centralise session handling
> `refactor(frontend): consolidate API clients and centralise session handling`

**Files affected:**
- `creator-app/src/lib/services/apiClient.js` — extended with `authenticatedFetch`/`authenticatedFetchJson` backward-compat aliases and 403 `X-Account-Status` (disabled/deleted) detection integrated directly into the `apiFetch` and `apiAxios` interceptors.
- `creator-app/src/lib/utils/apiClient.js` — **deleted**.
- `creator-app/src/lib/utils/sessionGuard.js` — simplified: removed `handleApiResponse` and `forceLogout`. Now only manages background polling using `apiFetch` instead of raw `fetch`.
- `creator-app/src/routes/+layout.svelte` — removed the side-effect import of `utils/apiClient`.
- `creator-app/src/routes/admin/+page.svelte` — migrated `import axios from 'axios'` + `authenticatedFetch` to `services/apiClient`.
- `creator-app/src/routes/org-admin/+page.svelte` — migrated `authenticatedFetch` from `utils/apiClient` to `services/apiClient`.

**Problems fixed:**
- Eliminated the race condition between the two sets of Axios interceptors (global instance vs isolated instance).
- Disabled-account detection (403) and session expiry (401) are now handled from a single point, guaranteeing automatic logout on any failing network request.
- Removed the `utils/apiClient` side-effect import that was mutating the global `axios` instance.

---

### PS-2 — Unify sessionManager and fix store leak on logout
> `fix(frontend): unify sessionManager and fix logout store leak`

**Files affected:**
- `ui/src/lib/session/sessionManager.js` — added hook system (`registerOnClearSession`). `clearCurrentSession()` now runs all registered callbacks after `user.logout()`. `ensureProfileLoaded()` improved: clears the session if the profile fetch fails.
- `ui/src/lib/index.js` — `clearCurrentSession`, `ensureProfileLoaded` and `registerOnClearSession` added to `@lamb/ui` public API.
- `creator-app/src/lib/session/sessionManager.js` — removed duplicate `clearCurrentSession` and `ensureProfileLoaded` implementations; now re-exported from `@lamb/ui`. `resetAllUserScopedStores`, `replaceSessionWithLoginData` and `replaceSessionWithToken` remain here as they are creator-app–specific.
- `creator-app/src/routes/+layout.svelte` — registered `resetAllUserScopedStores` as a callback in `onMount` via `registerOnClearSession`.

**Problems fixed:**
- **Real bug:** clicking "Logout" in the Nav called `clearCurrentSession()` from `@lamb/ui`, which only ran `user.logout()` without resetting creator-app stores (assistants, rubrics, templates, AAC tabs). A second user logging in could see the previous user's data.
- All logout paths (Nav button, 401 from apiClient, 403 disabled, polling) now automatically run `resetAllUserScopedStores`.
- Removed the duplicate `clearCurrentSession` and `ensureProfileLoaded` implementations that existed across the two `sessionManager.js` files.

---
## Files affected by this merge
### Auto-merged (no conflicts)
- `.gitignore`
- `Documentation/352-resilience-fixes-repro.md`
- `Documentation/lamb_architecture_v2.md`
- `backend/.env.example`
- `backend/config.py`
- `backend/creator_interface/library_manager_client.py`
- `backend/docker-entrypoint.py`
- `frontend/packages/creator-app/src/lib/components/AssistantsList.svelte`
- `frontend/packages/creator-app/src/lib/components/KnowledgeBasesList.svelte`
- `frontend/packages/creator-app/src/lib/components/aac/*`
- `frontend/packages/creator-app/src/lib/components/evaluaitor/RubricsList.svelte`
- `frontend/packages/creator-app/src/lib/components/libraries/LibraryDetail.svelte`
- `frontend/packages/creator-app/src/lib/components/modals/*`
- `frontend/packages/creator-app/src/lib/config.js`
- `frontend/packages/creator-app/src/lib/services/aacService.js`
- `frontend/packages/creator-app/src/lib/services/analyticsService.js`
- `frontend/packages/creator-app/src/lib/services/assistantService.js`
- `frontend/packages/creator-app/src/lib/services/knowledgeBaseService.js`
- `frontend/packages/creator-app/src/lib/services/libraryService.js`
- `frontend/packages/creator-app/src/lib/services/organizationService.js`
- `frontend/packages/creator-app/src/lib/services/rubricService.js`
- `frontend/packages/creator-app/src/lib/services/templateService.js`
- `frontend/packages/creator-app/src/lib/services/testService.js`
- `frontend/packages/creator-app/src/lib/session/sessionManager.js`
- `frontend/packages/creator-app/src/lib/stores/aacStore.svelte.js`
- `frontend/packages/creator-app/src/lib/stores/templateStore.js`
- `frontend/packages/creator-app/src/routes/agent/+page.svelte`
- `frontend/packages/creator-app/src/routes/agent/history/[id]/+page.svelte`
- `frontend/packages/creator-app/src/routes/assistants/+page.svelte`
### Resolved conflicts
- `ChatInterface.svelte` — accepted dev (chat history, streaming abort, isMounted)
- `Login.svelte` — merged: `authService.login` + `try/finally` from dev
- `PublishModal.svelte` — merged: `@lamb/ui` imports + `onDestroy` cleanup from dev
- `Signup.svelte` — merged: `authService.signup` + `try/finally` from dev
- `AssistantForm.svelte` — merged: `@lamb/ui` imports + `apiFetch` from dev
- `adminService.js` — accepted dev (uses `apiFetch`)
- `assistantStore.js` — accepted dev (defensive `Array.isArray()`)
- `+layout.js` — accepted dev (`try/catch` around `waitLocale()`)
- `+page.svelte` — merged: `@lamb/ui` imports + `apiFetch`/`sanitize` from dev
- `org-admin/+page.svelte` — merged: `apiAxios` from dev + `@lamb/ui` imports
- `Nav.svelte` — accepted HEAD (relative paths correct for `@lamb/ui`). Added `getConfig` as hardcoded `window.LAMB_CONFIG` read (TODO #7 to move to `@lamb/ui` properly)
- `userStore.js` — merged: `authService` from HEAD + validation from dev
### Deleted
- `frontend/svelte-app/package.json`
- `frontend/svelte-app/src/lib/services/authService.js`
- `frontend/svelte-app/static/config.js.sample`
### Added
- `frontend/packages/creator-app/src/lib/services/apiClient.js`
- `frontend/packages/ui/src/lib/utils/sanitize.js`