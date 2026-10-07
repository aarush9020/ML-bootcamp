(() => {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  const fileInput = $('file-input'), dropzone = $('dropzone'), fileLabel = $('file-label');
  const btnProcess = $('btn-process'), btnReset = $('btn-reset');
  const errorBanner = $('error-banner'), stepper = $('stepper'), statusMsg = $('status-msg');
  const results = $('results'), recordBox = $('record');

  let selectedFile = null, jobId = null, pollTimer = null, failures = 0;
  let rawText = '', refinedText = '';

  /* ---------- helpers ---------- */
  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.remove('hidden');
  }
  function clearError() { errorBanner.classList.add('hidden'); errorBanner.textContent = ''; }

  function setBusy(busy) {
    btnProcess.disabled = busy || !selectedFile;
    btnProcess.textContent = busy ? 'Processing...' : 'Process';
    fileInput.disabled = busy;
  }

  function setFile(file) {
    selectedFile = file;
    clearError();
    if (file) {
      const mb = (file.size / 1048576).toFixed(1);
      fileLabel.textContent = `${file.name} (${mb} MB)`;
      fileLabel.classList.add('has-file');
    } else {
      fileLabel.textContent = 'Select a meeting recording to begin';
      fileLabel.classList.remove('has-file');
    }
    btnProcess.disabled = !file;
  }

  function downloadText(name, text) {
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
    const a = el('a'); a.href = url; a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function setExportLinks(id) {
    for (const [btn, fmt] of [['dl-pdf', 'pdf'], ['dl-txt', 'txt'], ['dl-json', 'json']]) {
      const a = $(btn);
      if (id) { a.href = `/api/jobs/${id}/export/${fmt}`; a.classList.remove('disabled'); }
      else { a.href = '#'; a.classList.add('disabled'); }
    }
  }

  function resetUI() {
    clearTimeout(pollTimer);
    jobId = null; rawText = ''; refinedText = ''; failures = 0;
    setFile(null);
    fileInput.value = ''; fileInput.disabled = false;
    $('glossary').value = '';
    btnProcess.textContent = 'Process';
    clearError();
    stepper.classList.add('hidden'); statusMsg.classList.add('hidden');
    results.classList.add('hidden'); recordBox.classList.add('hidden');
    $('raw-transcript').replaceChildren(el('p', 'placeholder', 'Waiting for transcription...'));
    $('refined-transcript').replaceChildren(el('p', 'placeholder', 'Waiting for refinement...'));
    $('dl-raw').disabled = true; $('dl-refined').disabled = true;
    setExportLinks(null);
  }

  /* ---------- rendering ---------- */
  function renderStages(stages, message) {
    stepper.classList.remove('hidden');
    stepper.querySelectorAll('li').forEach((li) => {
      li.className = stages[li.dataset.stage] || 'pending';
    });
    if (message) { statusMsg.textContent = message; statusMsg.classList.remove('hidden'); }
  }

  const unspecifiedOr = (td, v) => {
    if (v === 'unspecified') td.append(el('span', 'unspecified', 'Unspecified'));
    else td.textContent = v;
  };

  function renderRecord(rec) {
    $('summary').textContent = rec.summary || 'No summary was generated.';

    const minutes = $('minutes'); minutes.replaceChildren();
    if (rec.minutes.length) {
      rec.minutes.forEach((m) => {
        const t = el('div', 'topic'); t.append(el('h4', '', m.topic));
        const ul = el('ul'); m.points.forEach((p) => ul.append(el('li', '', p)));
        t.append(ul); minutes.append(t);
      });
    } else minutes.append(el('p', 'empty', 'No minutes were generated.'));

    const dec = $('decisions'); dec.replaceChildren();
    if (rec.key_decisions.length) {
      const ol = el('ol'); rec.key_decisions.forEach((d) => ol.append(el('li', '', d))); dec.append(ol);
    } else dec.append(el('p', 'empty', 'No decisions were stated in the recording.'));

    const act = $('actions-table'); act.replaceChildren();
    if (rec.action_items.length) {
      const table = el('table'), head = el('tr');
      ['Task', 'Owner', 'Deadline'].forEach((h) => head.append(el('th', '', h)));
      table.append(head);
      rec.action_items.forEach((a) => {
        const tr = el('tr'), tdTask = el('td', '', a.task), tdOwner = el('td'), tdDue = el('td');
        unspecifiedOr(tdOwner, a.owner); unspecifiedOr(tdDue, a.deadline);
        tr.append(tdTask, tdOwner, tdDue); table.append(tr);
      });
      act.append(table);
    } else act.append(el('p', 'empty', 'No action items were stated in the recording.'));

    recordBox.classList.remove('hidden');
  }

  /* ---------- workflow ---------- */
  async function startProcessing() {
    if (!selectedFile) return;
    clearError();
    setBusy(true);

    const form = new FormData();
    form.append('file', selectedFile);
    form.append('glossary', $('glossary').value);

    try {
      const res = await fetch('/api/jobs', { method: 'POST', body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || 'The upload was rejected by the server.');
      jobId = data.job_id;
    } catch (e) {
      showError(e.message === 'Failed to fetch' ? 'Could not reach the server. Is the backend running?' : e.message);
      setBusy(false);
      return;
    }

    results.classList.remove('hidden');
    renderStages({ transcribe: 'running', refine: 'pending', extract: 'pending' }, 'Uploading complete. Starting...');
    poll();
  }

  async function poll() {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      const job = await res.json();
      if (!res.ok) throw new Error(job.detail || 'Lost track of the job.');
      failures = 0;

      renderStages(job.stages, job.message);
      if (job.raw_transcript && job.raw_transcript !== rawText) {
        rawText = job.raw_transcript;
        $('raw-transcript').textContent = rawText;
        $('dl-raw').disabled = false;
      }
      if (job.refined_transcript && job.refined_transcript !== refinedText) {
        refinedText = job.refined_transcript;
        $('refined-transcript').textContent = refinedText;
        $('dl-refined').disabled = false;
      }

      if (job.status === 'done') {
        renderRecord(job.record);
        setExportLinks(jobId);
        statusMsg.textContent = 'Done. Your meeting record is ready to download.';
        setBusy(false); fileInput.disabled = false;
        $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
      } else if (job.status === 'error') {
        showError(job.error || 'Processing failed.');
        statusMsg.classList.add('hidden');
        setBusy(false);
      } else {
        pollTimer = setTimeout(poll, 1000);
      }
    } catch (e) {
      if (++failures < 4) { pollTimer = setTimeout(poll, 1500); return; }
      showError(e.message === 'Failed to fetch' ? 'Lost connection to the server.' : e.message);
      setBusy(false);
    }
  }

  /* ---------- events ---------- */
  fileInput.addEventListener('change', () => setFile(fileInput.files[0] || null));
  ['dragenter', 'dragover'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.add('drag');
  }));
  ['dragleave', 'drop'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.remove('drag');
  }));
  dropzone.addEventListener('drop', (e) => {
    if (fileInput.disabled) return;
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });
  btnProcess.addEventListener('click', startProcessing);
  btnReset.addEventListener('click', resetUI);
  $('dl-raw').addEventListener('click', () => downloadText('raw-transcript.txt', rawText));
  $('dl-refined').addEventListener('click', () => downloadText('refined-transcript.txt', refinedText));

  fetch('/api/info').then((r) => r.json()).then((info) => {
    const ul = $('models-list'); ul.replaceChildren();
    [['Speech-to-text', info.models.speech_to_text],
     ['Language model 1 (transcript refinement)', info.models.refinement],
     ['Language model 2 (minutes, decisions, action items)', info.models.extraction]]
      .forEach(([k, v]) => { const li = el('li'); li.append(el('strong', '', k + ': '), document.createTextNode(v)); ul.append(li); });
    $('dz-sub').textContent = `English-language recordings: ${info.supported_extensions.join(' ')} (max ${info.max_upload_mb} MB)`;
  }).catch(() => { $('models-list').replaceChildren(el('li', '', 'Model information unavailable.')); });
})();
