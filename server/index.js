// ---------- ENV + CORE IMPORTS ----------
const path = require('path');

// Load .env from project root:  <root>/.env
require('dotenv').config({
  path: path.join(__dirname, '..', '.env'),
});

const fs = require('fs');
const fsp = fs.promises;

const express = require('express');
const session = require('express-session');
const cookieParser = require('cookie-parser');
const cors = require('cors');
const multer = require('multer');
const FormData = require('form-data');
const axios = require('axios');
const { Pool } = require('pg');

// helper that builds rich SUNEDU ZIPs
const { createAdminZip } = require('./adminZipHelper');

const {
  studentLogin,
  setStudentAccessToken,
  setStudentRefreshToken,
  studentFetch,
  adminLogin,
  adminGetStudent,
  adminGetCourseSchedules,
  adminGetTeachers,
  adminGetTeacherSchedule,
} = require('./uma');

const {
  PORT = 5000,
  SESSION_SECRET = 'change-this',
  VALIDATOR_URL: ENV_VALIDATOR_URL,
  UMA_DATABASE_URL,
  DATABASE_URL,
  POSTGRES_URL,
  SUPABASE_DB_URL,
} = process.env;

// Python validator URL (FastAPI)
const VALIDATOR_URL = ENV_VALIDATOR_URL || 'http://127.0.0.1:8000';

// ------------ Database (Supabase Postgres) ------------
const DB_URL =
  SUPABASE_DB_URL ||
  UMA_DATABASE_URL ||
  DATABASE_URL ||
  POSTGRES_URL ||
  '';

let DB_ENABLED = false;
let pool = null;

if (DB_URL) {
  const safeDbUrl = DB_URL.replace(/:\/\/([^:]+):[^@]+@/, '://$1:****@');
  console.log('[db] Using Postgres database at:', safeDbUrl);

  pool = new Pool({
    connectionString: DB_URL,
    ssl: { rejectUnauthorized: false }, // required for Supabase
  });

  pool.on('error', (err) => {
    console.error('[db] pool error', err);
  });

  DB_ENABLED = true;
} else {
  console.warn(
    '[db] No database URL configured. Falling back to submissions.json only.'
  );
}

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

// ---------- paths ----------
const ROOT_DIR = path.join(__dirname, '..');
// Python validator writes here: photo/photos/{approved|rejected}
const PHOTOS_ROOT = path.join(ROOT_DIR, 'photo', 'photos');
const SUBMISSIONS_PATH = path.join(ROOT_DIR, 'photo', 'submissions.json');

// directory for generated SUNEDU ZIP files
const ZIP_OUTPUT_DIR = path.join(ROOT_DIR, 'tmp_zips');
if (!fs.existsSync(ZIP_OUTPUT_DIR)) {
  fs.mkdirSync(ZIP_OUTPUT_DIR, { recursive: true });
}

// ---------- submissions helpers ----------
async function loadSubmissionsFromDb() {
  if (!DB_ENABLED || !pool) return [];

  const q = `
    select
      dni,
      codigo,
      name,
      email,
      facultad,
      carrera,
      category,
      issues,
      supabase_url,
      photo_filename,
      sunedu_status,
      updated_at
    from uma_submissions
    order by updated_at desc
  `;

  const { rows } = await pool.query(q);

  return rows.map((row) => {
    const issues = Array.isArray(row.issues)
      ? row.issues
      : row.issues
      ? row.issues
      : [];
    const category = row.category || 'approved';
    const photoUrl =
      row.supabase_url ||
      (row.photo_filename ? `/photos/${category}/${row.photo_filename}` : null);

    return {
      dni: row.dni,
      code: row.codigo,
      codigo: row.codigo,
      name: row.name,
      email: row.email,
      facultad: row.facultad,
      carrera: row.carrera,
      category,
      issues,
      supabase_url: row.supabase_url,
      photo_filename: row.photo_filename,
      suneduStatus: row.sunedu_status,
      updatedAt: row.updated_at,
      photoUrl,
    };
  });
}

async function loadSubmissionsFromFile() {
  try {
    const txt = await fsp.readFile(SUBMISSIONS_PATH, 'utf8');
    const parsed = JSON.parse(txt);
    if (Array.isArray(parsed)) return parsed;
    if (Array.isArray(parsed.submissions)) return parsed.submissions;
    return [];
  } catch (err) {
    if (err.code === 'ENOENT') return [];
    console.error('[submissions] read error:', err);
    return [];
  }
}

async function saveSubmissionsToFile(list) {
  try {
    await fsp.mkdir(path.dirname(SUBMISSIONS_PATH), { recursive: true });
    await fsp.writeFile(
      SUBMISSIONS_PATH,
      JSON.stringify({ submissions: list }, null, 2),
      'utf8'
    );
  } catch (err) {
    console.error('[submissions] write error:', err);
  }
}

async function loadSubmissions() {
  if (DB_ENABLED) {
    return loadSubmissionsFromDb();
  }
  return loadSubmissionsFromFile();
}

async function upsertSubmissionInDb(submission) {
  if (!DB_ENABLED || !pool) return;

  const issues = Array.isArray(submission.issues)
    ? submission.issues
    : submission.issues
    ? submission.issues
    : [];

  const suneduStatus = submission.suneduStatus || 'Pendiente';

  const q = `
    insert into uma_submissions (
      dni,
      codigo,
      name,
      email,
      facultad,
      carrera,
      category,
      issues,
      supabase_url,
      photo_filename,
      sunedu_status,
      updated_at
    ) values (
      $1, $2, $3, $4, $5, $6, $7,
      $8, $9, $10, $11, now()
    )
    on conflict (dni) do update set
      codigo = excluded.codigo,
      name = excluded.name,
      email = excluded.email,
      facultad = excluded.facultad,
      carrera = excluded.carrera,
      category = excluded.category,
      issues = excluded.issues,
      supabase_url = excluded.supabase_url,
      photo_filename = excluded.photo_filename,
      sunedu_status = excluded.sunedu_status,
      updated_at = now()
  `;

  const params = [
    submission.dni,
    submission.code || submission.codigo || null,
    submission.name || null,
    submission.email || null,
    submission.facultad || null,
    submission.carrera || submission.esp || null,
    submission.category || 'approved',
    issues,
    submission.supabase_url || null,
    submission.filename || submission.photo_filename || null,
    suneduStatus,
  ];

  await pool.query(q, params);
}

// Find the approved JPG for a given DNI:
//   photo/photos/approved/<dni>.jpg
function findApprovedPhotoByDni(dni) {
  if (!dni) return null;
  const dirApproved = path.join(PHOTOS_ROOT, 'approved');

  const jpg = path.join(dirApproved, `${dni}.jpg`);
  if (fs.existsSync(jpg)) return jpg;

  try {
    const files = fs.readdirSync(dirApproved);
    const hit = files.find((name) => name.startsWith(String(dni)));
    if (hit) return path.join(dirApproved, hit);
  } catch (err) {
    // dir may not exist yet
  }

  return null;
}

async function deletePhotoFile(absPath) {
  if (!absPath) return;
  try {
    const abs = path.resolve(absPath);
    const root = path.resolve(PHOTOS_ROOT);
    if (!abs.startsWith(root)) {
      console.warn('[delete] refused outside photos root:', abs);
      return;
    }
    await fsp.unlink(abs);
  } catch (err) {
    if (err.code !== 'ENOENT') console.warn('[delete] unlink error:', err);
  }
}

async function deleteSubmissionsInDb(dniList) {
  if (!DB_ENABLED || !pool || !Array.isArray(dniList) || !dniList.length) {
    return 0;
  }
  const q = `
    delete from uma_submissions
    where dni = any($1)
  `;
  const { rowCount } = await pool.query(q, [dniList]);
  return rowCount || 0;
}

async function markSuneduSentInDb(dniList) {
  if (!DB_ENABLED || !pool || !Array.isArray(dniList) || !dniList.length) {
    return 0;
  }
  const q = `
    update uma_submissions
    set sunedu_status = 'Enviado',
        updated_at = now()
    where dni = any($1)
  `;
  const { rowCount } = await pool.query(q, [dniList]);
  return rowCount || 0;
}

// ---------- middleware ----------
app.use(cors({ origin: true, credentials: true }));
app.use(express.json());
app.use(cookieParser());
app.use(
  session({
    secret: SESSION_SECRET,
    resave: false,
    saveUninitialized: false,
    cookie: { httpOnly: true, sameSite: 'lax' },
  })
);

// static assets
app.use(express.static(path.join(ROOT_DIR, 'public')));
app.use('/photos', express.static(PHOTOS_ROOT));
app.use('/downloads', express.static(ZIP_OUTPUT_DIR));

// ---------- STUDENT LOGIN ----------
app.post('/api/student/login', async (req, res) => {
  try {
    const { codigo, dni } = req.body;
    if (!codigo || !dni) {
      return res
        .status(400)
        .json({ ok: false, error: 'codigo and dni are required' });
    }

    const r = await studentLogin({ codigo, dni });

    const root = r.data || {};
    const data = root.data || root;
    const access = data.access_token || root.access_token || null;
    const refresh = data.refresh_token || root.refresh_token || null;

    if (!access) {
      return res.status(502).json({
        ok: false,
        error: 'UMA login did not return tokens',
        raw: root,
      });
    }

    setStudentAccessToken(req.session, access);
    setStudentRefreshToken(req.session, refresh);

    res.json({ ok: true, message: 'login ok' });
  } catch (e) {
    const status = e.response?.status || e.status || 500;
    res
      .status(status)
      .json({ ok: false, error: e.response?.data || e.message });
  }
});

// ---------- ADMIN LOGIN ----------
app.post('/api/admin/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) {
      return res
        .status(400)
        .json({ ok: false, error: 'email and password are required' });
    }

    const r = await adminLogin({ email, password });

    const root = r.data || {};
    const data = root.data || root;
    const access = data.access_token || root.access_token || null;

    if (!access) {
      return res.status(502).json({
        ok: false,
        error: 'UMA admin login did not return token',
        raw: root,
      });
    }

    req.session.adminAccessToken = access;
    res.json({ ok: true, message: 'admin login ok' });
  } catch (e) {
    const status = e.response?.status || e.status || 500;
    res
      .status(status)
      .json({ ok: false, error: e.response?.data || e.message });
  }
});

// ---------- STUDENT PROFILE ----------
app.post('/api/student/profile', async (req, res) => {
  try {
    const { code } = req.body;
    if (!code) {
      return res.status(400).json({ ok: false, error: 'code is required' });
    }
    const r = await adminGetStudent({ code });
    res.json({ ok: true, data: r.data });
  } catch (e) {
    const status = e.response?.status || e.status || 500;
    res
      .status(status)
      .json({ ok: false, error: e.response?.data || e.message });
  }
});

app.post('/api/student/course-schedules', async (req, res) => {
  try {
    const { code, period } = req.body;
    if (!code || !period) {
      return res
        .status(400)
        .json({ ok: false, error: 'code and period are required' });
    }
    const r = await adminGetCourseSchedules({ code, period });
    res.json({ ok: true, data: r.data });
  } catch (e) {
    const status = e.response?.status || e.status || 500;
    res
      .status(status)
      .json({ ok: false, error: e.response?.data || e.message });
  }
});

// ---------- ADMIN DATA ----------
app.post('/api/admin/student', async (req, res) => {
  try {
    const { code } = req.body;
    if (!code) {
      return res.status(400).json({ ok: false, error: 'code is required' });
    }
    const r = await adminGetStudent({ code });
    res.json({ ok: true, data: r.data });
  } catch (e) {
    const status = e.response?.status || e.status || 500;
    res
      .status(status)
      .json({ ok: false, error: e.response?.data || e.message });
  }
});

app.post('/api/admin/course-schedules', async (req, res) => {
  try {
    const { code, period } = req.body;
    if (!code || !period) {
      return res
        .status(400)
        .json({ ok: false, error: 'code and period are required' });
    }
    const r = await adminGetCourseSchedules({ code, period });
    res.json({ ok: true, data: r.data });
  } catch (e) {
    const status = e.response?.status || e.status || 500;
    res
      .status(status)
      .json({ ok: false, error: e.response?.data || e.message });
  }
});

app.post('/api/admin/teachers', async (req, res) => {
  try {
    const { period } = req.body;
    if (!period) {
      return res.status(400).json({ ok: false, error: 'period is required' });
    }
    const r = await adminGetTeachers({ period });
    res.json({ ok: true, data: r.data });
  } catch (e) {
    const status = e.response?.status || e.status || 500;
    res
      .status(status)
      .json({ ok: false, error: e.response?.data || e.message });
  }
});

app.post('/api/admin/teacher-schedule', async (req, res) => {
  try {
    const { dni, period } = req.body;
    if (!dni || !period) {
      return res
        .status(400)
        .json({ ok: false, error: 'dni and period are required' });
    }
    const r = await adminGetTeacherSchedule({ dni, period });
    res.json({ ok: true, data: r.data });
  } catch (e) {
    const status = e.response?.status || e.status || 500;
    res
      .status(status)
      .json({ ok: false, error: e.response?.data || e.message });
  }
});

// ---------- PHOTO VALIDATOR PROXY + LOG ----------
app.post('/validate', upload.single('image'), async (req, res) => {
  try {
    const file = req.file;
    const bodyFields = req.body || {};
    const dni = bodyFields.dni || 'unknown_user';
    const code = bodyFields.code || '';

    if (!file) {
      return res.status(400).json({ ok: false, issues: ['No file provided'] });
    }

    // ----- enrich with UMA data -----
    let name = bodyFields.name || '';
    let email = bodyFields.email || '';
    let esp = bodyFields.esp || '';
    let facultad = bodyFields.facultad || bodyFields.faculty || '';

    if (code && (!name || !email || !esp || !facultad)) {
      try {
        const r = await adminGetStudent({ code });
        const root = r.data || {};
        const s = root.data || root || {};

        const firstName = s.name || s.nombres || s.nombre || '';
        const lastName =
          s.lastname ||
          s.apellidos ||
          s.apellido ||
          [s.apellidoPaterno, s.apellidoMaterno].filter(Boolean).join(' ') ||
          '';
        const fullName = [firstName, lastName].filter(Boolean).join(' ');

        if (!name && fullName) name = fullName;

        if (!email) {
          email =
            s.email_institucional ||
            s.emailInstitucional ||
            s.email ||
            '';
        }

        if (!esp) {
          esp =
            s.carrera ||
            s.especialidad ||
            s.specialtyName ||
            s.schoolName ||
            '';
        }

        if (!facultad) {
          facultad =
            s.facultad ||
            s.faculty ||
            s.facultyName ||
            s.facultadNombre ||
            '';
        }
      } catch (err) {
        console.warn(
          '[validate] adminGetStudent failed for code',
          code,
          err.message || err
        );
      }
    }

    // ----- call Python validator -----
    const formData = new FormData();
    formData.append('image', file.buffer, {
      filename: file.originalname,
      contentType: file.mimetype || 'application/octet-stream',
    });
    formData.append('dni', dni);

    const url = `${VALIDATOR_URL}/validate`;
    console.log('[validate] calling validator at:', url);

    const response = await axios.post(url, formData, {
      headers: formData.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      validateStatus: () => true,
    });

    const data = response.data || {};

    // ----- log submission -----
    try {
      const ok = !!data.ok;
      const category = data.category || (ok ? 'approved' : 'rejected');
      const filename = data.filename || '';
      const relPath = (data.relative_path || '').toString();

      let photoUrl = '';
      if (filename) {
        photoUrl = `/photos/${category}/${filename}`;
      } else if (relPath) {
        const normalized = relPath.replace(/\\/g, '/');
        if (normalized.startsWith('photos/')) {
          const tail = normalized.slice('photos/'.length);
          photoUrl = `/photos/${tail}`;
        }
      }

      const now = new Date().toISOString();

      const submission = {
        dni,
        code,
        name,
        email,
        facultad,
        carrera: esp,
        esp,
        category,
        ok,
        photoUrl,
        filename,
        relative_path: relPath,
        issues: Array.isArray(data.issues) ? data.issues : [],
        data_url: data.data_url || null,
        supabase_url: data.supabase_url || null,
        suneduStatus: 'Pendiente',
        updatedAt: now,
      };

      if (DB_ENABLED) {
        await upsertSubmissionInDb(submission);
      } else {
        const list = await loadSubmissionsFromFile();
        const idxExisting = list.findIndex((s) => s.dni === dni);
        if (idxExisting >= 0) {
          list[idxExisting] = { ...list[idxExisting], ...submission };
        } else {
          submission.createdAt = now;
          list.push(submission);
        }
        await saveSubmissionsToFile(list);
      }
    } catch (err) {
      console.error('[submissions] log error:', err);
    }

    res.status(response.status || 200).json(data);
  } catch (err) {
    console.error('Validator proxy error:', err);
    res.status(500).json({
      ok: false,
      issues: ['Validation service error: ' + err.message],
    });
  }
});

// ---------- PHOTO AUTO-FIX PROXY (no log / no save) ----------
app.post('/fix-photo', upload.single('image'), async (req, res) => {
  try {
    const file = req.file;

    if (!file) {
      return res.status(400).json({ ok: false, issues: ['No file provided'] });
    }

    const formData = new FormData();
    formData.append('image', file.buffer, {
      filename: file.originalname || 'photo.jpg',
      contentType: file.mimetype || 'application/octet-stream',
    });

    const url = `${VALIDATOR_URL}/fix-photo`;
    console.log('[fix-photo] calling validator at:', url);

    const response = await axios.post(url, formData, {
      headers: formData.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      validateStatus: () => true,
    });

    const data = response.data || {};
    res.status(response.status || 200).json(data);
  } catch (err) {
    console.error('Fix-photo proxy error:', err);
    res.status(500).json({
      ok: false,
      issues: [
        'Error interno al intentar corregir la foto automáticamente.',
      ],
    });
  }
});

// ---------- ADMIN: list submissions ----------
app.get('/api/admin/submissions', async (_req, res) => {
  try {
    const list = await loadSubmissions();
    const approved = list.filter((s) => s.category === 'approved');
    const rejected = list.filter((s) => s.category !== 'approved');
    res.json({ ok: true, data: { approved, rejected } });
  } catch (err) {
    console.error('[submissions] admin list error:', err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ---------- ADMIN: generate ZIP ----------
app.post('/api/admin/generate-zip', async (req, res) => {
  try {
    const { dniList } = req.body || {};
    const list = await loadSubmissions();
    let selected = list.filter((s) => s.category === 'approved');

    if (Array.isArray(dniList) && dniList.length) {
      const dniSet = new Set(dniList.map(String));
      selected = selected.filter((s) => s.dni && dniSet.has(String(s.dni)));
    }

    console.log(
      '[zip] approved in storage:',
      list.filter((s) => s.category === 'approved').length
    );
    console.log('[zip] requested DNIs:', selected.map((s) => s.dni));

    if (!selected.length) {
      return res
        .status(400)
        .json({ ok: false, error: 'No hay estudiantes seleccionados.' });
    }

    const { zipPath, total, fileName } = await createAdminZip(selected, {
      outDir: ZIP_OUTPUT_DIR,
    });

    const publicUrl = `/downloads/${fileName}`;

    return res.json({
      ok: true,
      url: publicUrl,
      total,
      zipPath,
      file: fileName,
    });
  } catch (err) {
    console.error('[zip] unexpected error:', err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ---------- ADMIN: delete submissions ----------
app.post('/api/admin/delete-submissions', async (req, res) => {
  try {
    const { dniList } = req.body || {};
    if (!Array.isArray(dniList) || !dniList.length) {
      return res.status(400).json({ ok: false, error: 'dniList vacío.' });
    }

    const listBefore = await loadSubmissions();
    const toDelete = listBefore.filter(
      (s) => s.dni && dniList.includes(s.dni)
    );

    let deleted = 0;
    if (DB_ENABLED) {
      deleted = await deleteSubmissionsInDb(dniList);
    } else {
      const remaining = listBefore.filter(
        (s) => !(s.dni && dniList.includes(s.dni))
      );
      await saveSubmissionsToFile(remaining);
      deleted = toDelete.length;
    }

    // remove local photo files for those DNIs (best-effort)
    for (const s of toDelete) {
      const abs = findApprovedPhotoByDni(s.dni);
      if (abs) await deletePhotoFile(abs);
    }

    res.json({ ok: true, deleted });
  } catch (err) {
    console.error('[delete-submissions] error:', err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ---------- ADMIN: mark SUNEDU sent ----------
app.post('/api/admin/mark-sunedu-sent', async (req, res) => {
  try {
    const { dniList } = req.body || {};
    if (!Array.isArray(dniList) || !dniList.length) {
      return res.status(400).json({ ok: false, error: 'dniList vacío.' });
    }

    let updated = 0;

    if (DB_ENABLED) {
      updated = await markSuneduSentInDb(dniList);
    } else {
      const list = await loadSubmissionsFromFile();
      const now = new Date().toISOString();
      const updatedList = list.map((s) => {
        if (s.dni && dniList.includes(s.dni)) {
          updated += 1;
          return { ...s, suneduStatus: 'Enviado', updatedAt: now };
        }
        return s;
      });
      await saveSubmissionsToFile(updatedList);
    }

    res.json({ ok: true, updated });
  } catch (err) {
    console.error('[mark-sunedu-sent] error:', err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    validator: VALIDATOR_URL,
    photosRoot: PHOTOS_ROOT,
    dbEnabled: DB_ENABLED,
  });
});

app.listen(PORT, () => {
  console.log(`UMA proxy running on port ${PORT}`);
  console.log(`Validator URL configured as: ${VALIDATOR_URL}`);
  console.log(`PHOTOS_ROOT: ${PHOTOS_ROOT}`);
  console.log(`ZIP_OUTPUT_DIR: ${ZIP_OUTPUT_DIR}`);
  console.log(`DB_ENABLED: ${DB_ENABLED}`);
});
