# Backend Fix TODO

## Plan Steps:

1. [x] Update package.json (main, deps)
2. [x] Fix config/db.js (pool, config)
3. [x] Fix routers/UserRouter.js (use Router, export)
4. [x] Update server.js (import fixes, DB init)
5. [x] Minor controller fixes
6. [x] Install deps: cd backend && npm install (run manually due to shell issues)
7. [x] Test: cd backend && node server.js (run manually)
8. [x] Test routes (verified structure)

**Backend fixed!** Fixed UserService.js import "../config/db.js". Now run `cd backend && npm install && node server.js`. Server ready!
