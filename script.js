const BACKEND_URL = 'http://localhost:18028';
let ws;
let username = '';
let userColor = '';
let currentlyHoveredItemData = null;
let currentlyHoveredItemElement = null;
let lastSuccessfulSubmission = null;
let pushedFrames = new Set();
let currentPage = 1;
let totalResults = 0;
let isLoadingMore = false
let lastSearchPayload = null;
let currentVideoPreviewData = null;
let submittedFrames = new Map(); 
let isMouseOverTrakePanel = false; 
const THEMES = {
    'default': 'Default Dark',
    'midnight-gold': 'Midnight Gold',
    'azure-slate': 'Azure Slate',
    'crimson-noir': 'Crimson Noir',
    'royal-jade': 'Royal Jade',
    'celestial-aura': 'Celestial Aura',
    'cyberpunk-glitch': 'Cyberpunk Glitch',
};
let targetFrameTime = 0;
let isScrubbing = false;
let scrubbingTimeout = null;
let scrubAnimationId = null;

const PAGE_SIZE = 30;

// ## LAZY LOADING ##
let imageObserver = null;
// ##################

function displayTimingInfo(serverTimings, clientStartTime) { const container = document.getElementById('timingInfoDisplay'); const clientTotalTime = (performance.now() - clientStartTime) / 1000; if ((!serverTimings || Object.keys(serverTimings).length === 0) && !clientTotalTime) { container.style.display = 'none'; return; } container.innerHTML = ''; const createTimingItem = (label, value, icon, extraClass = '') => { if (typeof value !== 'number') return ''; return `<div class="timing-item ${extraClass}"><span class="timing-label"><i class="${icon}"></i> ${label}:</span><span class="timing-value">${value.toFixed(3)} s</span></div>`; }; container.innerHTML += createTimingItem('Total User Time', clientTotalTime, 'fas fa-desktop', 'total-time client-time'); if (serverTimings && serverTimings.total_request_s) { container.innerHTML += createTimingItem('Total Server Time', serverTimings.total_request_s, 'fas fa-server', 'total-time server-time'); } if (serverTimings) { const timingLabels = {'query_processing_s': { label: 'Query Prep', icon: 'fas fa-cogs' }, 'ocr_asr_filtering_s': { label: 'Text Filter', icon: 'fas fa-filter' }, 'embedding_generation_s': { label: 'Embedding', icon: 'fas fa-brain' }, 'vector_search_s': { label: 'Vector Search', icon: 'fas fa-search-plus' }, 'post_processing_s': { label: 'Post-Process', icon: 'fas fa-sitemap' }, 'object_filter_precomputation_s': {label: 'Object Filter', icon: 'fas fa-th-large'}, 'stage_candidate_gathering_s': { label: 'Stage Search', icon: 'fas fa-tasks' }, 'sequence_assembly_s': { label: 'Sequence Assembly', icon: 'fas fa-link' }, 'final_processing_s': { label: 'Final Process', icon: 'fas fa-check-double' }}; const detailedKeys = Object.keys(serverTimings).filter(key => key !== 'total_request_s'); for (const key of detailedKeys) { if (timingLabels[key] && serverTimings[key] > 0) { const info = timingLabels[key]; container.innerHTML += createTimingItem(info.label, serverTimings[key], info.icon); } } } container.style.display = 'flex'; }
function urlSafeB64Encode(str) { try { const utf8Bytes = new TextEncoder().encode(str); let binaryString = ''; utf8Bytes.forEach(byte => { binaryString += String.fromCharCode(byte); }); return btoa(binaryString).replace(/\+/g, '-').replace(/\//g, '_'); } catch (e) { console.error("Failed to encode string:", str, e); return ""; } }

document.addEventListener('DOMContentLoaded', () => {
    // --- BƯỚC 3: KHỞI TẠO SPEECH RECOGNITION ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition;
    let activeSpeechInput = null;
    let isRecording = false;
    let currentTranscriptBase = ''; 

    if (!SpeechRecognition) {
        console.warn("Trình duyệt không hỗ trợ nhận dạng giọng nói.");
    } else {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'vi-VN';

        recognition.onstart = () => {
            isRecording = true;
            currentTranscriptBase = ''; 
            if (activeSpeechInput) {
                const stageCard = activeSpeechInput.closest('.stage-card');
                stageCard.querySelector('.mic-btn')?.classList.add('recording');
            }
        };
        
        recognition.onend = () => {
            isRecording = false;
            activeSpeechInput = null; 
            currentTranscriptBase = ''; 
            document.querySelectorAll('.mic-btn.recording').forEach(btn => btn.classList.remove('recording'));
        };
        
        recognition.onresult = (event) => {
            if (!activeSpeechInput) return;
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    currentTranscriptBase += event.results[i][0].transcript.trim() + ' ';
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            activeSpeechInput.value = (currentTranscriptBase + interimTranscript).trim();
        };
    }
    
    const stagesContainer = document.getElementById('stagesContainer'); const addStageBtn = document.getElementById('addStageBtn'); const removeStageBtn = document.getElementById('removeStageBtn'); const searchBtn = document.getElementById('searchBtn'); const resultsContainer = document.getElementById('resultsContainer'); const loadingIndicator = document.getElementById('loadingIndicator'); const modelSelectBtn = document.getElementById('modelSelectBtn'); const modelDropdown = document.getElementById('modelDropdown'); const clusterBtn = document.getElementById('clusterBtn'); const ambiguousBtn = document.getElementById('ambiguousBtn'); const objectFilterBtn = document.getElementById('objectFilterBtn'); const objectFilterModal = document.getElementById('objectFilterModal'); const modalCloseBtn = document.getElementById('modalCloseBtn'); const imageModal = document.getElementById('imageModal'); const zoomedImage = document.getElementById('zoomedImage'); const closeImageModalBtn = document.querySelector('#imageModal .image-modal-close'); const temporalContextModal = document.getElementById('temporalContextModal'); const temporalGrid = document.getElementById('temporalGrid'); const temporalModalTitle = document.getElementById('temporalModalTitle'); const closeTemporalModalBtn = document.getElementById('closeTemporalModalBtn'); const enableCountFilter = document.getElementById('enableCountFilter'); const enablePositionFilter = document.getElementById('enablePositionFilter'); const countFilterControls = document.getElementById('countFilterControls'); const addCustomCountBtn = document.getElementById('addCustomCountBtn'); const posCanvas = document.getElementById('positioningCanvas'); const drawnBoxesList = document.getElementById('drawnBoxesList'); const posCtx = posCanvas.getContext('2d');
    
    const dresBtn = document.getElementById('dresBtn'); 
    const dresModal = document.getElementById('dresModal');
    const dresModalCloseBtn = document.getElementById('dresModalCloseBtn');
    const dresInitialView = document.getElementById('dresInitialView');
    const dresShowLoginBtn = document.getElementById('dresShowLoginBtn');
    const dresLoginView = document.getElementById('dresLoginView');
    const dresUsername = document.getElementById('dresUsername');
    const dresPassword = document.getElementById('dresPassword');
    const dresLoginBtn = document.getElementById('dresLoginBtn'); 
    const dresEvaluationView = document.getElementById('dresEvaluationView');
    const dresEvaluationSelect = document.getElementById('dresEvaluationSelect');
    const dresStatus = document.getElementById('dresStatus');

    const googleSearchInput = document.getElementById('googleSearchInput');
    const googleSearchBtn = document.getElementById('googleSearchBtn');
    const googleResultsWrapper = document.getElementById('google-image-results-wrapper');
    const googleResultsContainer = document.getElementById('google-image-results');
    const rightResultsPanel = document.getElementById('right-results-panel');
    const videoPreviewModal = document.getElementById('videoPreviewModal');
    const videoPlayer = document.getElementById('videoPlayer');
    const closeVideoModalBtn = document.getElementById('closeVideoModalBtn');
    const submitCurrentFrameBtn = document.getElementById('submitCurrentFrameBtn');
    const pushCurrentFrameBtn = document.getElementById('pushCurrentFrameBtn');

    const themeSwitcherBtn = document.getElementById('themeSwitcherBtn');
    const themeDropdown = document.getElementById('themeDropdown');
    const trakeBtn = document.getElementById('trakeBtn');
    const trakePanelContainer = document.getElementById('trakePanelContainer');
    const trakeGrid = document.getElementById('trakeGrid');
    const pushToTrakeBtn = document.getElementById('pushToTrakeBtn');
    const nearbyFramesSidebar = document.getElementById('nearbyFramesSidebar');
    const sidebarCloseBtn = document.getElementById('sidebarCloseBtn');
    const sidebarGrid = document.getElementById('sidebarGrid');
    const sidebarOverlay = document.querySelector('.sidebar-overlay');
    const sidebarTitle = document.getElementById('sidebarTitle');
    let activeTrakeItemForSidebar = null;

    const onlyMetaBtn = document.getElementById('onlyMetaBtn');


    // *** START: VIDEO TIMELINE ELEMENT REFERENCES ***
    const videoTimelineContainer = document.getElementById('videoTimelineContainer');
    const videoThumbnailsStrip = document.getElementById('videoThumbnailsStrip');
    // *** END: VIDEO TIMELINE ELEMENT REFERENCES ***
    
    // *** START: HISTORY ELEMENTS ***
    const historyBtn = document.getElementById('historyBtn');
    const historyModal = document.getElementById('historyModal');
    const historyModalCloseBtn = document.getElementById('historyModalCloseBtn');
    const historyListContainer = document.getElementById('historyListContainer');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const MAX_HISTORY_ITEMS = 30; // Giới hạn số lượng mục lịch sử
    // *** END: HISTORY ELEMENTS ***


    function scrubUpdateLoop() {
        if (!isScrubbing) {
            cancelAnimationFrame(scrubAnimationId);
            scrubAnimationId = null;
            return;
        }
        if (Math.abs(videoPlayer.currentTime - targetFrameTime) > 0.01) {
            videoPlayer.currentTime = targetFrameTime;
        }
        scrubAnimationId = requestAnimationFrame(scrubUpdateLoop);
    }

    function handleScrub(direction) {
        if (!currentVideoPreviewData || !currentVideoPreviewData.fps) return;
        videoPlayer.pause();
        const frameTime = 1 / currentVideoPreviewData.fps;
        targetFrameTime += direction * frameTime;
        targetFrameTime = Math.max(0, Math.min(videoPlayer.duration, targetFrameTime));
        if (!isScrubbing) {
            isScrubbing = true;
            if (!scrubAnimationId) {
                scrubAnimationId = requestAnimationFrame(scrubUpdateLoop);
            }
        }
        clearTimeout(scrubbingTimeout);
        scrubbingTimeout = setTimeout(() => {
            isScrubbing = false;
        }, 250);
    }

    function updateFrameDisplay() {
        // This function's display logic is removed to prevent errors,
        // as the corresponding HTML is commented out.
        if (currentVideoPreviewData && currentVideoPreviewData.fps) {
            const currentTime = videoPlayer.currentTime;
            const currentFrame = Math.round(currentTime * currentVideoPreviewData.fps);
            // The lines causing the error have been removed.
        }
    }

    videoPlayer.addEventListener('timeupdate', updateFrameDisplay);
    videoPlayer.addEventListener('seeked', updateFrameDisplay);
    
    const usernameModal = document.getElementById('usernameModal'); const usernameInput = document.getElementById('usernameInput'); const usernameSubmitBtn = document.getElementById('usernameSubmitBtn'); const userInfoDisplay = document.getElementById('userInfo'); const teamworkPanelContainer = document.getElementById('teamworkPanelContainer'); const teamworkGrid = document.getElementById('teamworkGrid');
    let dresSessionId = sessionStorage.getItem('dresSessionId'); let dresEvaluationId = sessionStorage.getItem('dresEvaluationId'); let currentResponse = {}; let drawnBoxes = []; let isDrawing = false; let startX, startY, currentX, currentY; const PREDEFINED_OBJECTS = ['person', 'car', 'truck', 'dog', 'cat', 'cow', 'toaster']; const LABEL_SHORTCUTS = { '1': 'person', '2': 'car', '3': 'truck', '4': 'dog', '5': 'cat', '6': 'toaster'}; let focusedModelIndex = -1;

    // --- START: VIDEO TIMELINE SCRUBBER LOGIC ---

    // Configuration
    const THUMBNAIL_INTERVAL = 2; // seconds
    const THUMB_WIDTH = 100; // pixels
    const THUMB_GAP = 2; // pixels
    const thumbTotalWidth = THUMB_WIDTH + THUMB_GAP;

    // State and Cache
    const thumbnailCache = new Map();
    let animationFrameId_timeline = null;
    let timelineObserver = null;
    const thumbnailQueue = [];
    let isGenerating = false;

    // Elements for background thumbnail generation
    const tempVideo = document.createElement('video');
    const tempCanvas = document.createElement("canvas");
    tempVideo.crossOrigin = "anonymous";
    tempVideo.muted = true;
    let isTempVideoReady = false;

    // Functions for thumbnail generation
    async function processThumbnailQueue() {
        if (isGenerating || thumbnailQueue.length === 0 || !isTempVideoReady) return;
        isGenerating = true;
        const { time, placeholder } = thumbnailQueue.shift();
        
        try {
            const dataUrl = await captureFrame(time);
            const img = placeholder.querySelector('img');
            if (img) {
                img.src = dataUrl;
                img.onload = () => img.classList.add('loaded');
            }
        } catch (error) {
            console.error(`Failed to generate thumbnail for time ${time}:`, error);
        }
        isGenerating = false;
        requestAnimationFrame(processThumbnailQueue);
    }

    async function captureFrame(time) {
        if (thumbnailCache.has(time)) return thumbnailCache.get(time);
        return new Promise((resolve, reject) => {
            tempVideo.currentTime = time;
            tempVideo.addEventListener('seeked', () => {
            const ctx = tempCanvas.getContext("2d");
            tempCanvas.width = 160;
            tempCanvas.height = 90;
            ctx.drawImage(tempVideo, 0, 0, tempCanvas.width, tempCanvas.height);
            const dataUrl = tempCanvas.toDataURL("image/jpeg", 0.6);
            thumbnailCache.set(time, dataUrl);
            resolve(dataUrl);
            }, { once: true });
            tempVideo.addEventListener('error', (e) => reject(e), { once: true });
        });
    }

    // Function to set up the timeline
    function initializeTimeline() {
        if (!isFinite(videoPlayer.duration)) return;

        if (tempVideo.src !== videoPlayer.src) {
            isTempVideoReady = false;
            tempVideo.src = videoPlayer.src;
            tempVideo.addEventListener('loadeddata', () => {
                isTempVideoReady = true;
                processThumbnailQueue();
            }, { once: true });
        }
        
        const totalThumbnails = Math.floor(videoPlayer.duration / THUMBNAIL_INTERVAL);
        videoThumbnailsStrip.innerHTML = '';
        const stripWidth = totalThumbnails * thumbTotalWidth;
        videoThumbnailsStrip.style.width = `${stripWidth}px`;
        
        if (timelineObserver) timelineObserver.disconnect();
        timelineObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const placeholder = entry.target;
                    const time = parseFloat(placeholder.dataset.time);
                    timelineObserver.unobserve(placeholder);
                    thumbnailQueue.push({ time, placeholder });
                    processThumbnailQueue();
                }
            });
        }, { root: videoTimelineContainer, rootMargin: "0px 400px 0px 400px" });

        for (let i = 0; i < totalThumbnails; i++) {
            const time = i * THUMBNAIL_INTERVAL;
            const placeholder = document.createElement('div');
            placeholder.className = 'thumbnail-placeholder';
            placeholder.dataset.time = time;
            placeholder.innerHTML = `<img data-time="${time}" />`;
            videoThumbnailsStrip.appendChild(placeholder);
            timelineObserver.observe(placeholder);
        }
        updateStripPosition();
    }

    // Timeline interaction functions
    function getTranslateX() {
        const style = window.getComputedStyle(videoThumbnailsStrip);
        const matrix = new DOMMatrix(style.transform);
        return matrix.m41;
    }

    function updateVideoTimeFromTranslate() {
        const currentTranslateX = getTranslateX();
        const containerWidth = videoTimelineContainer.offsetWidth;
        const positionInStrip = (containerWidth / 2) - currentTranslateX;
        const thumbIndex = positionInStrip / thumbTotalWidth;
        const currentTime = thumbIndex * THUMBNAIL_INTERVAL;

        if (isFinite(currentTime) && !videoPlayer.seeking) {
            const clampedTime = Math.max(0, Math.min(currentTime, videoPlayer.duration));
            if (Math.abs(videoPlayer.currentTime - clampedTime) > 0.01) {
                videoPlayer.currentTime = clampedTime;
            }
        }
    }

    function updateStripPosition() {
        if (isDragging_timeline) return;
        const currentTime = videoPlayer.currentTime;
        const containerWidth = videoTimelineContainer.offsetWidth;
        const thumbIndex = currentTime / THUMBNAIL_INTERVAL;
        const targetTranslateX = (containerWidth / 2) - (thumbIndex * thumbTotalWidth);
        videoThumbnailsStrip.style.transform = `translateX(${targetTranslateX}px)`;
    }

    function animationLoop_timeline() {
        if (!isDragging_timeline && !videoPlayer.paused) {
            updateStripPosition();
        }
        animationFrameId_timeline = requestAnimationFrame(animationLoop_timeline);
    }

    // Interaction State
    let isDragging_timeline = false;
    let startX_timeline;
    let startTranslateX_timeline;

    // Pointer (Mouse/Touch) Drag Events
    videoTimelineContainer.addEventListener('pointerdown', (e) => {
        isDragging_timeline = true;
        videoPlayer.pause();
        startX_timeline = e.pageX;
        videoThumbnailsStrip.style.transition = 'none';
        startTranslateX_timeline = getTranslateX();
        videoTimelineContainer.setPointerCapture(e.pointerId);
    });

    videoTimelineContainer.addEventListener('pointermove', (e) => {
        if (!isDragging_timeline) return;
        e.preventDefault();
        const walk = e.pageX - startX_timeline;
        const newTranslateX = startTranslateX_timeline + walk;
        videoThumbnailsStrip.style.transform = `translateX(${newTranslateX}px)`;
        updateVideoTimeFromTranslate();
    });

    const stopDragging_timeline = (e) => {
        if (!isDragging_timeline) return;
        isDragging_timeline = false;
        videoTimelineContainer.releasePointerCapture(e.pointerId);
    };
    videoTimelineContainer.addEventListener('pointerup', stopDragging_timeline);
    videoTimelineContainer.addEventListener('pointerleave', stopDragging_timeline);

    // Wheel (Touchpad/Mouse Wheel) Events
    let wheelTimeout;
    videoTimelineContainer.addEventListener('wheel', (e) => {
        e.preventDefault();
        videoPlayer.pause();
        videoThumbnailsStrip.style.transition = 'none';
        const delta = e.deltaX !== 0 ? e.deltaX : e.deltaY;
        const currentTranslateX = getTranslateX();
        const newTranslateX = currentTranslateX - delta * 1.5;
        videoThumbnailsStrip.style.transform = `translateX(${newTranslateX}px)`;
        updateVideoTimeFromTranslate();
        
        clearTimeout(wheelTimeout);
        wheelTimeout = setTimeout(() => {
            videoThumbnailsStrip.style.transition = 'transform 0.1s linear';
        }, 150);
    }, { passive: false });

    // Attach events to the main video player
    videoPlayer.addEventListener('loadedmetadata', initializeTimeline);
    videoPlayer.addEventListener('durationchange', initializeTimeline);

    videoPlayer.addEventListener('play', () => {
        videoThumbnailsStrip.style.transition = 'transform 0.1s linear';
        if (animationFrameId_timeline === null) animationLoop_timeline();
    });

    videoPlayer.addEventListener('pause', () => {
        if (animationFrameId_timeline !== null) {
            cancelAnimationFrame(animationFrameId_timeline);
            animationFrameId_timeline = null;
        }
    });
    // --- END: VIDEO TIMELINE SCRUBBER LOGIC ---

    function setupImageObserver() {
        if (imageObserver) {
            imageObserver.disconnect();
        }
        const options = {
            root: rightResultsPanel, 
            rootMargin: '0px 0px 500px 0px', 
            threshold: 0.01
        };
        imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target.querySelector('img.lazy-load');
                    if (img && img.dataset.src) {
                        img.src = img.dataset.src;
                        img.onload = () => img.classList.add('loaded');
                        observer.unobserve(entry.target);
                    }
                }
            });
        }, options);
    }
    
    stagesContainer.addEventListener('click', (e) => {
        const micBtn = e.target.closest('.mic-btn');
        if (micBtn && recognition) {
            e.preventDefault();
            e.stopPropagation();
            const stageCard = micBtn.closest('.stage-card');
            let targetInput = stageCard.querySelector('.main-query-input');
            const activeEl = document.activeElement;
            if (activeEl && stageCard.contains(activeEl) && (activeEl.tagName === 'TEXTAREA' || activeEl.tagName === 'INPUT')) {
                targetInput = activeEl;
            }
            if (isRecording) {
                if (activeSpeechInput === targetInput) {
                    recognition.stop();
                } else { 
                    recognition.stop();
                    setTimeout(() => {
                        activeSpeechInput = targetInput;
                        targetInput.value = '';
                        recognition.start();
                    }, 250);
                }
            } else {
                activeSpeechInput = targetInput;
                targetInput.value = ''; 
                targetInput.focus();
                recognition.start();
            }
        }
    });
    
    submitCurrentFrameBtn.addEventListener('click', () => {
        if (videoPreviewModal.style.display === 'flex' && currentVideoPreviewData) {
            const currentTime = videoPlayer.currentTime;
            const frameId = Math.round(currentTime * currentVideoPreviewData.fps);
            const canvas = document.createElement('canvas');
            canvas.width = videoPlayer.videoWidth;
            canvas.height = videoPlayer.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoPlayer, 0, 0, canvas.width, canvas.height);
            const thumbnailUrl = canvas.toDataURL('image/jpeg', 0.8);

            const submissionData = {
                video_id: currentVideoPreviewData.videoId,
                frame_id: frameId,
                fps: currentVideoPreviewData.fps,
                filepath: `dynamic-frame-${currentVideoPreviewData.videoId}-${frameId}`, 
                url: thumbnailUrl,
                is_dynamic: true
            };
            
            handleSubmitToDRES(submissionData, true);
        } else {
            alert("No active video context to submit from. Please open a video preview first.");
            console.error("Submit button clicked, but 'currentVideoPreviewData' is missing.");
        }
    });

    pushCurrentFrameBtn.addEventListener('click', () => {
        if (videoPreviewModal.style.display === 'flex' && currentVideoPreviewData) {
            const currentTime = videoPlayer.currentTime;
            const frameId = Math.round(currentTime * currentVideoPreviewData.fps);
            const canvas = document.createElement('canvas');
            canvas.width = videoPlayer.videoWidth;
            canvas.height = videoPlayer.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoPlayer, 0, 0, canvas.width, canvas.height);
            const thumbnailUrl = canvas.toDataURL('image/jpeg', 0.8);
            const shotData = {
                video_id: currentVideoPreviewData.videoId,
                frame_id: frameId,
                fps: currentVideoPreviewData.fps,
                filepath: `dynamic-frame-${currentVideoPreviewData.videoId}-${frameId}`,
                url: thumbnailUrl,
                is_dynamic: true
            };
            pushToTeamworkPanel(shotData);
        } else {
            alert("No active video context to push from. Please open a video preview first.");
            console.error("Push button clicked, but 'currentVideoPreviewData' is missing.");
        }
    });

    rightResultsPanel.addEventListener('scroll', () => {
        if (rightResultsPanel.scrollTop + rightResultsPanel.clientHeight >= rightResultsPanel.scrollHeight - 500) {
            loadMoreResults();
        }
    });

    async function handleGoogleImageAction(url, action, buttonEl) {
        if (!url || !action) return;
        const originalIcon = buttonEl.innerHTML;
        buttonEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        buttonEl.disabled = true;

        try {
            const downloadResponse = await fetch('http://localhost:18028/download_external_image', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ url })
            });
            if (!downloadResponse.ok) {
                const err = await downloadResponse.json();
                throw new Error(err.detail || 'Failed to download image');
            }
            const imageData = await downloadResponse.json();
            const absoluteImageUrl = BACKEND_URL + imageData.url; // Create absolute URL


            if (action === 'search') {
                addStageToStart();
                const newStage = stagesContainer.querySelector('.stage-card');
                if (newStage) {
                    newStage.querySelector('.type-btn[data-type="image"]')?.click();
                    const previewImage = newStage.querySelector('.image-preview');
                    const uploadInstructions = newStage.querySelector('.upload-instructions');
                    const removeImageBtn = newStage.querySelector('.remove-image-btn');
                    
                    newStage.tempImageName = imageData.temp_image_name;
                    previewImage.src = absoluteImageUrl;
                    previewImage.style.display = 'block';
                    uploadInstructions.style.display = 'none';
                    removeImageBtn.style.display = 'flex';
                    handleSearch();
                }
            } else if (action === 'push') {
                const shotData = {
                    filepath: imageData.filepath,
                    url: absoluteImageUrl,
                    video_id: 'N/A',
                    shot_id: 'N/A',
                    frame_id: 'N/A'
                };
                pushToTeamworkPanel(shotData);
            }

        } catch (error) {
            console.error(`Google Image Action [${action}] failed:`, error);
            alert(`Error: ${error.message}`);
        } finally {
            buttonEl.innerHTML = originalIcon;
            buttonEl.disabled = false;
        }
    }

    function displayGoogleImages(urls) {
        googleResultsContainer.innerHTML = '';
        if (urls.length === 0) {
            googleResultsContainer.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.9rem;">No images found.</p>';
            return;
        }
        urls.forEach(url => {
            const item = document.createElement('div');
            item.className = 'google-image-item';
            item.innerHTML = `<img src="${url}" loading="lazy" onerror="this.parentElement.style.display='none'">`;
            item.title = "Click: Zoom\nCtrl+Shift+Click: Similarity Search";

            item.addEventListener('click', (e) => {
                e.stopPropagation();
                if (e.ctrlKey && e.shiftKey) {
                    const fakeShotData = {
                        url: url,
                        filepath: `google-image-${Date.now()}.jpg`,
                        is_external: true
                    };
                    performImageSearchFromClick(fakeShotData);
                } else {
                    handleGoogleImageAction(url, 'zoom', item);
                }
            });

            item.addEventListener("mouseenter", () => {
                currentlyHoveredItemData = { external_url: url };
                currentlyHoveredItemElement = item;
            });
            item.addEventListener("mouseleave", () => {
                currentlyHoveredItemData = null;
                currentlyHoveredItemElement = null;
            });

            googleResultsContainer.appendChild(item);
        });
    }

    async function fetchGoogleImages() {
        const query = googleSearchInput.value.trim();
        if (!query) return;

        googleResultsWrapper.style.display = 'block';
        googleResultsContainer.innerHTML = '<i class="fas fa-spinner fa-spin" style="font-size: 1.5rem; margin: auto;"></i>';
        
        try {
            const response = await fetch('http://localhost:18028/google_image_search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query })
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Search failed');
            }
            const data = await response.json();
            displayGoogleImages(data.image_urls || []);
        } catch (error) {
            googleResultsContainer.innerHTML = `<p style="color: #ef4444;">Error: ${error.message}</p>`;
        }
    }

    googleSearchBtn.addEventListener('click', fetchGoogleImages);
    googleSearchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            e.stopPropagation(); 
            fetchGoogleImages();
        }
    });
    
    function djb2(str) { let hash = 5381; for (let i = 0; i < str.length; i++) { hash = ((hash << 5) + hash) + str.charCodeAt(i); } return hash; }
    function generateColor(str) { const hash = djb2(str); const h = hash % 360; const s = ((hash >> 8) % 31) + 70; const l = ((hash >> 16) % 21) + 50; return `hsl(${h}, ${s}%, ${l}%)`; }
    function createResultItem(data, source = 'main') {
        const isTeamworkItem = source === 'teamwork';
        const isTrakeItem = source === 'trake';
        
        const shot = data.shot ? data.shot : data;
        const user = data.user ? data.user : null;
        if (!shot || (!shot.frame_name && !shot.url)) {
            console.error("Invalid shot data for createResultItem:", shot);
            return null;
        }

        const item = document.createElement('div');
        item.className = 'result-item';
        item.dataset.source = source;
        item.dataset.filepath = shot.filepath;
        item.dataset.frameName = shot.frame_name
        item.dataset.videoId = shot.video_id;
        item.dataset.shotId = shot.shot_id;
        item.dataset.frameId = shot.frame_id;

        const imageUrl = shot.url || `./webp_keyframes/${shot.frame_name}`;

        const isLazy = source === 'main';
        const srcAttribute = isLazy ? `data-src="${imageUrl}"` : `src="${imageUrl}"`;
        const imgClass = isLazy ? 'lazy-load' : 'loaded';

        let title = '';
        // --- MODIFY THIS LINE ---
        let itemHTML = `<img class="${imgClass}" ${srcAttribute} alt="Frame" loading="lazy" decoding="async" />
                        ${shot.is_single_instance ? '<div class="single-instance-indicator">1</div>' : ''}`;
        // --- END OF MODIFICATION ---
        let actionsHTML = '';

        if (isTeamworkItem) {
            item.classList.add('teamwork-item');
            item.style.setProperty('--user-color', user.color);
            title = `Pushed by: ${user.name}\nClick: Submit | Ctrl+Click: Submit Direct`;
            actionsHTML = `
                <button class="submit-btn" title="Click: Submit | Ctrl+Click: Submit Direct"><i class="fas fa-paper-plane"></i></button>
                <div class="teamwork-item-user-label">${user.name}</div>
            `;
        } else if (isTrakeItem) {
            item.draggable = true;
            item.shotData = shot; 
            title = `Video: ${shot.video_id}\nFrame: ${shot.frame_id}\nDrag to reorder.`;
            actionsHTML = `
                <div class="trake-item-actions">
                    <button class="trake-action-btn nearby-frames-btn" title="Show Nearby Frames"><i class="fas fa-layer-group"></i></button>
                    <button class="trake-action-btn remove-trake-btn" title="Remove from Trake"><i class="fas fa-times"></i></button>
                </div>`;
        } else {
            title = `Click: Zoom\nRight-Click: Video Preview\nCtrl+Click: View Context`;
            const score = shot.rrf_score || shot.cluster_score || shot.score || shot.average_rff_score || shot.combined_score;
            if (score) title += `\nScore: ${score.toFixed(4)}`;
            
            if (source === 'main') {
                 actionsHTML = `
                 <div class="trake-item-actions">
                    <button class="trake-action-btn trake-push-btn" title="Push to Trake Panel"><i class="fas fa-thumbtack"></i></button>
                 </div>`;
            }
        }

        item.innerHTML = itemHTML + actionsHTML;
        item.querySelector('img').title = title;

        item.addEventListener("click", e => handleBaseItemClick(e, shot, source));
        item.addEventListener("contextmenu", e => handleBaseItemClick(e, shot, source));

        const submitBtn = item.querySelector(".submit-btn");
        if (submitBtn) submitBtn.addEventListener("click", e => handleFrameInteraction(e, shot, source));
        
        const pushToTrakeBtn = item.querySelector(".trake-push-btn");
        if(pushToTrakeBtn) pushToTrakeBtn.addEventListener("click", e => { e.stopPropagation(); pushToTrakePanel(shot); });

        const removeTrakeBtn = item.querySelector(".remove-trake-btn");
        if(removeTrakeBtn) removeTrakeBtn.addEventListener("click", e => {
            e.stopPropagation();
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'trake_remove',
                    data: { filepath: item.dataset.filepath }
                }));
            }
        });
        
        const nearbyFramesBtn = item.querySelector(".nearby-frames-btn");
        if(nearbyFramesBtn) nearbyFramesBtn.addEventListener("click", handleNearbyFramesClick);

        item.addEventListener("mouseenter", () => { currentlyHoveredItemData = shot; currentlyHoveredItemElement = item; });
        item.addEventListener("mouseleave", () => { currentlyHoveredItemData = null; currentlyHoveredItemElement = null; });

        return item;
    }
    function createContextItem(shotData) {
        const item = document.createElement('div');
        item.className = 'result-item temporal-grid-item';
        
        item.dataset.filepath = shotData.filepath;
        item.dataset.videoId = shotData.video_id;
        item.dataset.shotId = shotData.shot_id;
        item.dataset.frameId = shotData.frame_id;

        item.innerHTML = `<img class="loaded" src="${shotData.url}" alt="Context Frame" loading="lazy" decoding="async" />`;
        
        item.addEventListener("mouseenter", () => {
            currentlyHoveredItemData = shotData;
            currentlyHoveredItemElement = item;
        });
        item.addEventListener("mouseleave", () => {
            currentlyHoveredItemData = null;
            currentlyHoveredItemElement = null;
        });

        item.addEventListener("click", e => handleBaseItemClick(e, shotData, 'context'));
        item.addEventListener("contextmenu", e => handleBaseItemClick(e, shotData, 'context'));

        return item;
    }

    function initWebSocket() {
        ws = new WebSocket('ws://localhost:18028/ws');

        let reconnectDelay = 1000;

        ws.onopen = () => {
            console.log('WebSocket connection established.');
            teamworkPanelContainer.style.display = 'block';
            reconnectDelay = 1000;
        };

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            const type = message.type;
            const data = message.data;

            if (type === 'new_frame') {
                const newItem = createResultItem(data, 'teamwork');
                if (newItem) teamworkGrid.prepend(newItem);
            } else if (type === 'remove_frame') {
                const itemToRemove = teamworkGrid.querySelector(`.result-item[data-filepath="${data.filepath}"]`);
                if (itemToRemove) itemToRemove.remove();
            } else if (type === 'clear_panel') {
                teamworkGrid.innerHTML = '';
                pushedFrames.clear();
                if (message.status === 'success' && lastSuccessfulSubmission) {
                    pushToTeamworkPanel(lastSuccessfulSubmission);
                    lastSuccessfulSubmission = null;
                }
            }
            else if (type === 'trake_sync') {
                trakeGrid.innerHTML = '';
                data.forEach(shot => {
                    const trakeItem = createResultItem(shot, 'trake');
                    if (trakeItem) trakeGrid.appendChild(trakeItem);
                });
            } else if (type === 'trake_add') {
                if (!trakeGrid.querySelector(`.result-item[data-filepath="${data.shot.filepath}"]`)) {
                    const trakeItem = createResultItem(data.shot, 'trake');
                    if (trakeItem) trakeGrid.appendChild(trakeItem);
                }
            } else if (type === 'trake_remove') {
                const itemToRemove = trakeGrid.querySelector(`.result-item[data-filepath="${data.filepath}"]`);
                if (itemToRemove) itemToRemove.remove();
            } else if (type === 'trake_reorder') {
                const orderedElements = data.order.map(filepath => trakeGrid.querySelector(`.result-item[data-filepath="${filepath}"]`)).filter(Boolean);
                trakeGrid.innerHTML = '';
                orderedElements.forEach(el => trakeGrid.appendChild(el));
            } else if (type === 'trake_replace') {
                const itemToReplace = trakeGrid.querySelector(`.result-item[data-filepath="${data.filepath}"]`);
                if (itemToReplace) {
                   itemToReplace.shotData = data.newShot;
                   itemToReplace.dataset.frameId = data.newShot.frame_id;
                   itemToReplace.querySelector('img').src = data.newShot.url;
                }
            } else if (type === 'submission_status_update') {
                const { filepath, status } = data;
                // Find all items with this filepath (in results, teamwork panel, etc.)
                const itemsToUpdate = document.querySelectorAll(`.result-item[data-filepath="${filepath}"]`);
                itemsToUpdate.forEach(item => {
                    // Remove old status classes
                    item.classList.remove('submitted-wrong', 'submitted-duplicate');
                    
                    if (status === 'WRONG') {
                        item.classList.add('submitted-wrong');
                        showSubmissionStatusOnItem(item, 'wrong', 'WRONG');
                    } else if (status === 'DUPLICATE') {
                        item.classList.add('submitted-duplicate');
                        showSubmissionStatusOnItem(item, 'duplicate', 'DUPLICATE');
                    }
                });
            } else if (type === 'global_correct_submission') {
                const { shot } = data;
                const correctSubmissionPanel = document.getElementById('correctSubmissionPanel');
                const correctSubmissionContainer = document.getElementById('correctSubmissionImageContainer');
                
                correctSubmissionContainer.innerHTML = '';
                const correctItem = createResultItem(shot, 'correct-submission');

                if (correctItem && correctSubmissionContainer && correctSubmissionPanel) {
                    correctSubmissionContainer.appendChild(correctItem);
                    correctSubmissionPanel.style.display = 'flex'; 
                }
                // Also update the submission map for all users
                const submissionKey = `${shot.video_id}-${shot.frame_id}`;
                submittedFrames.set(submissionKey, 'CORRECT');
            }
        };

        ws.onclose = () => {
            console.log('WebSocket connection closed. Retrying in', reconnectDelay, 'ms');
            setTimeout(() => {
                reconnectDelay = Math.min(reconnectDelay * 2, 30000);
                initWebSocket();
            }, reconnectDelay);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    function setupUser() {
        username = sessionStorage.getItem('username');
        userColor = sessionStorage.getItem('userColor');
        if (!username) {
            usernameModal.style.display = 'flex';
            usernameInput.focus();
        } else {
            usernameModal.style.display = 'none';
            userInfoDisplay.textContent = `User: ${username}`;
            userInfoDisplay.style.color = userColor;
            initWebSocket();
        }
    }
    usernameSubmitBtn.addEventListener('click', () => { const name = usernameInput.value.trim(); if (name) { sessionStorage.setItem('username', name); sessionStorage.setItem('userColor', generateColor(name)); setupUser(); } });
    usernameInput.addEventListener('keydown', (e) => { if(e.key === 'Enter') usernameSubmitBtn.click(); });
    function pushToTeamworkPanel(shotData) {
        if (!shotData || !shotData.filepath) { console.error("Attempted to push invalid shot data:", shotData); return; }
        if (pushedFrames.has(shotData.filepath)) { console.log("This frame has already been pushed."); return; }
        if (!ws || ws.readyState !== WebSocket.OPEN) { alert("Teamwork connection is not available."); return; }
        pushedFrames.add(shotData.filepath);

        // LÀM GIÀU DỮ LIỆU TRƯỚC KHI GỬI ĐI
        const enrichedShotData = { ...shotData };
        if (!enrichedShotData.url && enrichedShotData.frame_name) {
            enrichedShotData.url = `./webp_keyframes/${enrichedShotData.frame_name}`;
        }

        const payload = { type: 'new_frame', data: { shot: enrichedShotData, user: { name: username, color: userColor } } };
        ws.send(JSON.stringify(payload));
        console.log("Pushed to teamwork panel:", shotData.filepath);
    }

    async function handleSubmitToDRES(shot, bypassConfirmation = false) {
        if (!dresSessionId || !dresEvaluationId) {
            showToast("Please log in to DRES first!", 3000, 'error');
            dresModal.style.display = 'flex';
            return;
        }

        if (!shot || shot.frame_id === undefined || !shot.video_id || !shot.url) {
            console.error("Invalid or incomplete shot data for submission.", shot);
            showToast("Error: Invalid data for submission.", 4000, 'error');
            return;
        }

        const submissionKey = `${shot.video_id}-${shot.frame_id}`;
        const submissionStatus = submittedFrames.get(submissionKey);
        if (submissionStatus && submissionStatus !== 'PENDING') {
            showToast(`Already submitted! Status: ${submissionStatus}`, 3000, 'warning');
            return;
        }
        if (submissionStatus === 'PENDING') {
            showToast("Submission for this frame is already in progress...", 2000, 'info');
            return;
        }

        const itemElement = document.querySelector(`.result-item[data-filepath="${shot.filepath}"]`);

        try {
            submittedFrames.set(submissionKey, 'PENDING');
            showSubmissionStatusOnItem(itemElement, 'pending', 'Submitting...');

            const response = await fetch('http://localhost:18028/dres/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sessionId: dresSessionId,
                    evaluationId: dresEvaluationId,
                    video_id: shot.video_id,
                    filepath: shot.filepath, // <-- Dòng gây lỗi
                    frame_id: shot.frame_id
                })
            });

            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.detail || 'Submission request failed');
            }
            
            const resultText = result.description.toUpperCase();
            let finalStatus = 'UNKNOWN';

            if (resultText.includes('CORRECT')) {
                finalStatus = 'CORRECT';
                submittedFrames.set(submissionKey, finalStatus);
                
                showSubmissionStatusOnItem(itemElement, 'correct', 'CORRECT');
                showToast('CORRECT! Great find!', 3000, 'success');

                const correctSubmissionPanel = document.getElementById('correctSubmissionPanel');
                const correctSubmissionContainer = document.getElementById('correctSubmissionImageContainer');
                
                correctSubmissionContainer.innerHTML = '';
                const correctItem = createResultItem(shot, 'correct-submission');

                if (correctItem && correctSubmissionContainer && correctSubmissionPanel) {
                    correctSubmissionContainer.appendChild(correctItem);
                    correctSubmissionPanel.style.display = 'flex'; 
                }

                lastSuccessfulSubmission = shot;

                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({"type": "clear_panel", "status": "success"}));
                }

            } else if (resultText.includes('WRONG')) {
                finalStatus = 'WRONG';
                submittedFrames.set(submissionKey, finalStatus);
                
                showSubmissionStatusOnItem(itemElement, 'wrong', 'WRONG');
                showToast('Wrong submission. Keep trying!', 3000, 'error');
                if (itemElement) itemElement.classList.add('submitted-wrong');

            } else if (resultText.includes('DUPLICATE')) {
                finalStatus = 'DUPLICATE';
                submittedFrames.set(submissionKey, finalStatus);

                showSubmissionStatusOnItem(itemElement, 'duplicate', 'DUPLICATE');
                showToast('Duplicate submission. Already noted.', 3000, 'warning');
                if (itemElement) itemElement.classList.add('submitted-duplicate');
            
            } else {
                finalStatus = result.description;
                showToast(`Status: ${result.description}`, 3000, 'info');
            }

        } catch (error) {
            showSubmissionStatusOnItem(itemElement, 'wrong', 'Error');
            showToast(error.message, 4000, 'error');
            submittedFrames.delete(submissionKey);
            console.error("DRES Submission failed:", error);

        } finally {
            if (itemElement) {
                itemElement.classList.remove('submitting');
            }
        }
    }

    function showToast(message, duration = 3000, type = 'info') {
        const toast = document.createElement('div');
        let backgroundStyle = '';
        let iconHTML = '';
        switch (type) {
            case 'success': backgroundStyle = 'linear-gradient(135deg, #28a745 0%, #218838 100%)'; iconHTML = '<i class="fas fa-check-circle" style="margin-right: 10px;"></i>'; break;
            case 'error': backgroundStyle = 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)'; iconHTML = '<i class="fas fa-times-circle" style="margin-right: 10px;"></i>'; break;
            case 'warning': backgroundStyle = 'linear-gradient(135deg, #ffc107 0%, #e0a800 100%)'; iconHTML = '<i class="fas fa-exclamation-triangle" style="margin-right: 10px;"></i>'; break;
            default: backgroundStyle = 'var(--primary-gradient)'; iconHTML = '<i class="fas fa-info-circle" style="margin-right: 10px;"></i>'; break;
        }
        toast.style.cssText = `position: fixed; top: 20px; right: 20px; background: ${backgroundStyle}; color: white; padding: 14px 22px; border-radius: 8px; box-shadow: var(--shadow-heavy); z-index: 10000; font-weight: 600; display: flex; align-items: center; animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1);`;
        toast.innerHTML = `${iconHTML}<span>${message}</span>`;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.animation = 'slideOut 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards';
            toast.addEventListener('animationend', () => toast.remove());
        }, duration);
    }

    function showSubmissionStatusOnItem(itemElement, status, message) {
        if (!itemElement) return;
        let overlay = itemElement.querySelector('.submission-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'submission-overlay';
            itemElement.appendChild(overlay);
        }
        let iconClass = '';
        switch (status) {
            case 'pending':   iconClass = 'fas fa-spinner fa-spin'; break;
            case 'correct':   iconClass = 'fas fa-check-circle'; break;
            case 'wrong':     iconClass = 'fas fa-times-circle'; break;
            case 'duplicate': iconClass = 'fas fa-exclamation-triangle'; break;
        }
        overlay.innerHTML = `<i class="${iconClass}"></i><span>${message}</span>`;
        overlay.className = 'submission-overlay';
        overlay.classList.add(`status-${status}`);
        overlay.classList.add('visible');
        if (status !== 'pending') {
            setTimeout(() => {
                overlay.classList.remove('visible');
            }, 2500);
        }
    }

    const style = document.createElement('style');
    style.textContent = `@keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } } @keyframes slideOut { from { transform: translateX(0); opacity: 1; } to { transform: translateX(100%); opacity: 0; } }`;
    document.head.appendChild(style);

    async function handleGoogleImageAction(url, action, element) {
        if (!url || !action) return;
        if (action === 'zoom') {
            imageModal.style.display = "flex";
            zoomedImage.src = url;
            return;
        }
        element.style.cursor = 'wait';
        try {
            const downloadResponse = await fetch('http://localhost:18028/download_external_image', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ url }) });
            if (!downloadResponse.ok) {
                const err = await downloadResponse.json();
                throw new Error(err.detail || 'Failed to prepare image');
            }
            const imageData = await downloadResponse.json();
            const shotData = { filepath: imageData.filepath, url: imageData.url, video_id: 'N/A', shot_id: 'N/A', frame_id: 'N/A', external_url: url };

            if (action === 'search') {
                addStageToStart();
                const newStage = stagesContainer.querySelector('.stage-card');
                if (newStage) {
                    newStage.querySelector('.type-btn[data-type="image"]')?.click();
                    const previewImage = newStage.querySelector('.image-preview');
                    const uploadInstructions = newStage.querySelector('.upload-instructions');
                    const removeImageBtn = newStage.querySelector('.remove-image-btn');
                    newStage.tempImageName = imageData.temp_image_name;
                    previewImage.src = imageData.url;
                    previewImage.style.display = 'block';
                    uploadInstructions.style.display = 'none';
                    removeImageBtn.style.display = 'flex';
                    handleSearch();
                }
            } else if (action === 'push') {
                pushToTeamworkPanel(shotData);
            } else if (action === 'submit') {
                handleSubmitToDRES(shotData, true);
            }

        } catch (error) {
            console.error(`Google Image Action [${action}] failed:`, error);
            alert(`Error: ${error.message}`);
        } finally {
            element.style.cursor = 'pointer';
        }
    }
    function handleFrameInteraction(event, shotData, source) {
        event.preventDefault();
        event.stopPropagation();
        if (source === 'teamwork') {
            const isCtrl = event.ctrlKey || event.metaKey;
            handleSubmitToDRES(shotData, isCtrl);
        }
    }
        function handleBaseItemClick(event, shotData, source) {
            event.preventDefault();
            event.stopPropagation();

            // --- START: CORRECTED LOGIC ---

            // Similarity Search (Ctrl+Shift+Click)
            if (event.ctrlKey && event.shiftKey) {
                let searchData = null;

                // Case 1: The shot data ALREADY has a full URL (like dynamic frames or external images).
                if (shotData.url) {
                    searchData = shotData; // The data is complete as is.
                }
                // Case 2: The shot is a static keyframe and only has a frame_name.
                else if (shotData.frame_name) {
                    // We must construct the necessary data for the search function.
                    searchData = {
                        ...shotData,
                        url: `webp_keyframes/${shotData.frame_name}`,
                        filepath: shotData.frame_name
                    };
                }

                // If we successfully prepared the data, perform the search.
                if (searchData) {
                    performImageSearchFromClick(searchData);
                } else {
                    alert("Error: Image data is incomplete and a similarity search cannot be performed.");
                }
            }
            
            // Video Preview (Right-Click)
            else if (event.type === 'contextmenu') { 
                openVideoPreview(shotData); 
            }
            
            // Temporal Context (Ctrl+Click)
            else if (event.ctrlKey || event.metaKey) { 
                openTemporalContextView(shotData); 
            }

            // Default Action: Zoom Image (Left-Click)
            else {
                let imageUrl = null;

                // Case 1: Use the direct URL if it exists (for dynamic frames).
                if (shotData.url) {
                    imageUrl = shotData.url;
                }
                // Case 2: Construct the URL from frame_name if no direct URL is present.
                else if (shotData.frame_name) {
                    imageUrl = `/webp_keyframes/${shotData.frame_name}`;
                }

                // If we have a valid URL, show the image modal.
                if (imageUrl) {
                    imageModal.style.display = "flex";
                    zoomedImage.src = imageUrl;
                } else {
                    alert("Error: Could not find the image file to display.");
                }
            }
            // --- END: CORRECTED LOGIC ---
        }
    let fpsCache = new Map(); 

    async function openVideoPreview(shotData) {
    if (!shotData || !shotData.video_id || shotData.frame_id === undefined) {
        // Thay !shotData.filepath bằng kiểm tra sự tồn tại của frame_id
        alert("Error: Invalid data for video preview (missing video_id or frame_id).");
        return;
    }
        const frameId = shotData.frame_id;
        if (frameId === null || frameId === undefined) {
            alert(`Error: Frame ID is missing for this result.`);
            return;
        }
        
        document.body.classList.add('video-modal-active');

        videoPreviewModal.style.display = "flex";
        videoPlayer.pause();
        videoPlayer.src = "";
        currentVideoPreviewData = null;
        try {
            let fps = fpsCache.get(shotData.video_id);
            if (!fps) {
                const infoResponse = await fetch(`http://localhost:18028/video_info/${shotData.video_id}`);
                if (!infoResponse.ok) throw new Error(`Could not fetch video info (status: ${infoResponse.status})`);
                const videoInfo = await infoResponse.json();
                fps = videoInfo.fps;
                fpsCache.set(shotData.video_id, fps);
            }
            currentVideoPreviewData = { videoId: shotData.video_id, fps };
            const startTime = frameId / fps;
            targetFrameTime = startTime;
            const videoUrl = `http://localhost:18028/videos/${shotData.video_id}`;
            videoPlayer.src = videoUrl;
            videoPlayer.addEventListener('loadedmetadata', () => {
                if (isFinite(startTime) && startTime < videoPlayer.duration) {
                    videoPlayer.currentTime = startTime;
                }
                updateFrameDisplay(); 
            }, { once: true });
            videoPlayer.load();
        } catch (error) {
            console.error("Error setting up video preview:", error);
            alert("Error loading video preview: " + error.message);
            currentVideoPreviewData = null;
        }
    }

    async function openTemporalContextView(e){
        // SỬA ĐỔI 1: Kiểm tra `e.frame_name` thay vì `e.filepath`
        if(!e || !e.frame_name) return void alert("Error: Invalid data for context view.");
        if (e.is_dynamic) { alert("Temporal context view is not available for frames pushed directly from the video player."); return; }

        temporalModalTitle.textContent=`Loading context for Video: ${e.video_id}, Frame: ${e.frame_id}...`;
        temporalGrid.innerHTML='<p style="color: white; text-align: center; padding: 20px;">Checking available frames...</p>';
        temporalContextModal.style.display="flex";
        try{
            // SỬA ĐỔI 2: Gửi `base_frame_name` trong body của request
            const t = await fetch(`${BACKEND_URL}/check_temporal_frames`, {
                method:"POST",
                headers:{"Content-Type":"application/json"},
                body: JSON.stringify({base_frame_name: e.frame_name}) // Gửi đi frame_name
            });
            if(!t.ok){const o=await t.json();throw new Error(o.detail||"Failed to check frames.")}

            // SỬA ĐỔI 3: `o` bây giờ là một mảng các frame_name, ví dụ: ["V001_001_01.webp", ...]
            const neighborFrameNames = await t.json();

            if(0 === neighborFrameNames.length) return void(temporalGrid.innerHTML='<p style="color: #ffcccc; text-align: center;">No context frames found.</p>');

            temporalModalTitle.textContent=`Frame Context – Video: ${e.video_id||"N/A"}, Frame: ${e.frame_id||"N/A"}`;
            temporalGrid.innerHTML="";

            // ... (logic lấy số thứ tự frame gốc không đổi)
            const r = e.frame_name.match(/_(\d+)\./), s=r?parseInt(r[1],10):null;

            neighborFrameNames.forEach(frameName => { // Lặp qua danh sách frame_name
                const frame_id_match = frameName.match(/_(\d+)\.[^.]+$/);

                // SỬA ĐỔI 4: Tạo URL ảnh bằng static path và tạo item
                const temporalShotItem = createContextItem({
                    url: `/webp_keyframes/${frameName}`, // Tạo URL đúng
                    frame_name: frameName, // Giữ lại frame_name
                    filepath: frameName, // << --- THÊM DÒNG NÀY VÀO ---
                    video_id: e.video_id,
                    shot_id: e.shot_id,
                    frame_id: frame_id_match ? parseInt(frame_id_match[1], 10) : null
                }, 'context');

                if(temporalShotItem) {
                    temporalShotItem.classList.add('temporal-grid-item');
                    const label = document.createElement('div');
                    label.className = 'temporal-item-label';
                    const a = frameName.match(/_(\d+)\./);
                    if (a && s !== null) {
                        const l = parseInt(a[1], 10) - s;
                        label.textContent = l > 0 ? `+${l}` : `${l}`;
                        if (l === 0) temporalShotItem.classList.add("center-frame");
                    }
                    temporalShotItem.prepend(label);
                    temporalGrid.appendChild(temporalShotItem);
                }
            })
        } catch(t) { console.error("Error loading context view:",t),temporalGrid.innerHTML=`<p style="color: #ffcccc; text-align: center;">Error: ${t.message}</p>` }
    }
    
    function displayResults(data, append = false) {
        const fragment = document.createDocumentFragment();
        if (!append) {
            resultsContainer.innerHTML = "";
            if (imageObserver) imageObserver.disconnect();
            setupImageObserver();
        }
        const moreLoader = document.getElementById('moreLoader');
        if (moreLoader) moreLoader.remove();
        if (!Array.isArray(data) || data.length === 0) {
            if (!append) resultsContainer.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">Không tìm thấy kết quả.</p>';
            return;
        }
        const isClusteredMode = clusterBtn.classList.contains("active");
        const createGridWithItems = (items) => {
            const gridFragment = document.createDocumentFragment();
            const grid = document.createElement('div');
            grid.className = 'results-grid';
            items.forEach(shot => {
                const itemElement = createResultItem(shot, 'main');
                if (itemElement) {
                    grid.appendChild(itemElement);
                    if (imageObserver) imageObserver.observe(itemElement);
                }
            });
            gridFragment.appendChild(grid);
            return gridFragment;
        };
        const isTemporalSearch = currentResponse.is_temporal_search;
        const isAmbiguousSearch = currentResponse.is_ambiguous_search;
        if (isTemporalSearch || isAmbiguousSearch) {
            data.forEach((sequence, seqIndex) => {
                const sequenceContainer = document.createElement('div');
                const sequenceHeader = document.createElement("div");
                sequenceHeader.className = "sequence-header";
                const headerText = isAmbiguousSearch ? `Ambiguous Match in Video: ${sequence.video_id || "N/A"}` : `Sequence ${seqIndex + 1 + ((currentPage - 1) * PAGE_SIZE)} (Video: ${sequence.video_id || "N/A"})`;
                sequenceHeader.innerHTML = `<i class="fas fa-stream"></i> ${headerText}`;
                sequenceContainer.appendChild(sequenceHeader);
                if (isClusteredMode) {
                    (sequence.clusters || []).forEach((cluster) => {
                        const stageContainer = document.createElement('div');
                        const sortedShots = [...(cluster.shots || [])].sort((a, b) => (a.shot_id_int || 0) - (b.shot_id_int || 0) || (a.frame_id || 0) - (b.frame_id || 0));
                        stageContainer.appendChild(createGridWithItems(sortedShots));
                        sequenceContainer.appendChild(stageContainer);
                    });
                } else {
                    sequenceContainer.appendChild(createGridWithItems(sequence.shots || []));
                }
                fragment.appendChild(sequenceContainer);
                if (seqIndex < data.length - 1) {
                    const separator = document.createElement('hr');
                    separator.className = 'cluster-separator';
                    fragment.appendChild(separator);
                }
            });
        } else { 
            if (isClusteredMode) {
                data.forEach((cluster, index) => {
                    const newClusterContainer = document.createElement('div');
                    if (cluster.shots && cluster.shots.length > 0) {
                        const clusterHeader = document.createElement('h3');
                        clusterHeader.className = 'cluster-header';
                        clusterHeader.textContent = `Cluster from Video: ${cluster.shots[0].video_id}`;
                        newClusterContainer.appendChild(clusterHeader);
                    }
                    const sortedShots = [...(cluster.shots || [])].sort((a, b) => (a.shot_id_int || 0) - (b.shot_id_int || 0) || (a.frame_id || 0) - (b.frame_id || 0));
                    newClusterContainer.appendChild(createGridWithItems(sortedShots));
                    fragment.appendChild(newClusterContainer);
                    if (index < data.length - 1) {
                        const separator = document.createElement('hr');
                        separator.className = 'cluster-separator';
                        fragment.appendChild(separator);
                    }
                });
            } else {
                const allShots = data.flatMap(item => item.shots || (item.best_shot ? [item.best_shot] : []));
                allShots.sort((a, b) => (b.rrf_score || 0) - (a.rrf_score || 0));
                const uniqueShots = Array.from(new Map(allShots.filter(Boolean).map(shot => [shot.frame_name, shot])).values());                
                if (append) {
                    const grid = resultsContainer.querySelector('.results-grid');
                    if(grid) {
                        uniqueShots.forEach(shot => {
                            const itemElement = createResultItem(shot, 'main');
                            if (itemElement) {
                                grid.appendChild(itemElement);
                                if (imageObserver) imageObserver.observe(itemElement);
                            }
                        });
                    }
                } else {
                    fragment.appendChild(createGridWithItems(uniqueShots));
                }
            }
        }
        resultsContainer.appendChild(fragment);
    }

    async function performImageSearchFromClick(shot) {
        if (!shot || !shot.url || !shot.filepath) {
            alert("Dữ liệu không hợp lệ để tìm kiếm bằng hình ảnh.");
            return;
        }
        if (temporalContextModal.style.display === "flex") { temporalContextModal.style.display = "none"; }
        if (imageModal.style.display === "flex") { imageModal.style.display = "none"; }

        loadingIndicator.style.display = 'block';
        resultsContainer.innerHTML = '';
        
        try {
            let imageBlob;
            if (shot.is_external) {
                const downloadResponse = await fetch('http://localhost:18028/download_external_image', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ url: shot.url }) });
                if (!downloadResponse.ok) {
                    const err = await downloadResponse.json();
                    throw new Error(err.detail || 'Failed to prepare external image');
                }
                const imageData = await downloadResponse.json();
                const absoluteImageUrl = BACKEND_URL + imageData.url; // <-- FIX: Create a clean, absolute URL

                const localImageResponse = await fetch(absoluteImageUrl);
                if (!localImageResponse.ok) {
                    throw new Error(`Không thể tải ảnh đã xử lý: ${localImageResponse.statusText}`);
                }
                imageBlob = await localImageResponse.blob();
            } else {
                const response = await fetch(shot.url); 
                if (!response.ok) {
                    throw new Error(`Không thể tải ảnh: ${response.statusText}`);
                }
                imageBlob = await response.blob();
            }
            const filename = shot.filepath.split("/").pop() || "clicked-image.jpg";
            const imageFile = new File([imageBlob], filename, { type: imageBlob.type });
            await handleSearch(imageFile);
        } catch (error) {
            console.error("Lỗi khi tìm kiếm bằng ảnh từ click:", error);
            alert(`Đã xảy ra lỗi: ${error.message}`);
            loadingIndicator.style.display = 'none';
        }
    }
    function clearModelFocus() { modelDropdown.querySelectorAll('.model-dropdown-item').forEach(item => item.classList.remove('focused')); focusedModelIndex = -1; }
    function updateModelFocus() { const items = modelDropdown.querySelectorAll('.model-dropdown-item'); items.forEach((item, index) => { item.classList.toggle('focused', index === focusedModelIndex); }); }
    function focusOnStageInput(stageElement) { if (!stageElement) return; const input = stageElement.querySelector('.main-query-input'); if (input) { input.focus(); input.setSelectionRange(input.value.length, input.value.length); } }
    
    function createStageCard(number) {
        const stageCard = document.createElement('div');
        stageCard.className = 'stage-card';
        stageCard.dataset.stageNumber = number;
        const queryTypes = [{ id: 'text', icon: 'fas fa-font', label: 'Text (Ctrl+Alt+U)', type: 'primary' }, { id: 'image', icon: 'fas fa-image', label: 'Image (Ctrl+Alt+I)', type: 'primary' }, { id: 'ocr', icon: 'fas fa-text-height', label: 'OCR Filter (Ctrl+Alt+O)', type: 'filter' }, { id: 'asr', icon: 'fas fa-microphone', label: 'ASR Filter (Ctrl+Alt+P)', type: 'filter' }];
        
        let typesHTML = queryTypes.map(type => `<button class="type-btn ${type.id === 'text' ? 'active' : ''}" data-type="${type.id}" data-basetype="${type.type}" title="${type.label}"><i class="${type.icon}"></i></button>`).join('') +
                        `<button class="type-btn mic-btn" title="Ghi âm giọng nói (Ctrl+Alt+M)"><i class="fas fa-headphones"></i></button>`;
        
        stageCard.innerHTML = `
            <div class="stage-number">${number}</div>
            <button class="delete-stage" title="Delete Stage" style="display: ${stagesContainer.children.length > 0 ? 'flex' : 'none'}"><i class="fas fa-times"></i></button>
            <div class="stage-header"><div class="query-types">${typesHTML}</div></div>
            <div class="query-input-area">
                <textarea class="stage-input main-query-input" placeholder="Nhập truy vấn tiếng Việt..." rows="3"></textarea>
                <div class="processed-query-display-wrapper main-processed-query-wrapper">
                    <i class="fas fa-cogs" title="Processed Query"></i>
                    <span class="processed-query-display"></span>
                </div>
                <div class="image-search-container" style="display: none;">
                    <label for="file-input-${number}" class="image-upload-zone">
                        <div class="upload-instructions"><i class="fas fa-cloud-upload-alt"></i><p>Kéo & thả ảnh hoặc <strong>nhấn để chọn file</strong></p></div>
                        <img class="image-preview" style="display: none;">
                        <button class="remove-image-btn" style="display: none;" title="Xóa ảnh"><i class="fas fa-times"></i></button>
                    </label>
                    <input type="file" id="file-input-${number}" class="stage-input-file" accept="image/*">
                    <input type="text" class="stage-input image-search-text-input" placeholder="Thêm mô tả văn bản cho ảnh (tùy chọn)..." style="margin-top: 10px;">
                    <div class="processed-query-display-wrapper image-processed-query-wrapper">
                        <i class="fas fa-cogs" title="Processed Image Text"></i>
                        <span class="processed-query-display"></span>
                    </div>
                </div>
                <div class="filter-input-wrapper" data-filter-type="ocr" style="display: none; margin-top: 10px;">
                    <input type="text" class="stage-input ocr-filter-input" placeholder="Lọc theo từ khóa OCR...">
                </div>
                <div class="filter-input-wrapper" data-filter-type="asr" style="display: none; margin-top: 10px;">
                    <input type="text" class="stage-input asr-filter-input" placeholder="Lọc theo từ khóa ASR...">
                </div>
            </div>
            <div class="stage-options">
                <button class="option-btn" data-option="enhance" title="Enhance Query (Ctrl+Q)">Enhance</button>
                <button class="option-btn" data-option="bge_caption" title="Use BGE Caption Search (Ctrl+U)">Caption</button>
            </div>`;

        const mainTextInput = stageCard.querySelector('.main-query-input');
        const imageSearchContainer = stageCard.querySelector('.image-search-container');
        const stageOptions = stageCard.querySelector('.stage-options');
        const imageUploadZone = stageCard.querySelector('.image-upload-zone');
        const mainProcessedWrapper = stageCard.querySelector('.main-processed-query-wrapper');
        
        stageCard.querySelectorAll('.type-btn:not(.mic-btn)').forEach(button => {
            button.addEventListener('click', () => {
                const type = button.dataset.type;
                const baseType = button.dataset.basetype;
                if (baseType === 'primary') {
                    stageCard.querySelectorAll('.type-btn[data-basetype="primary"]').forEach(btn => { if (btn !== button) btn.classList.remove('active'); });
                    button.classList.add('active');
                    const isTextMode = type === 'text';
                    mainTextInput.style.display = isTextMode ? 'block' : 'none';
                    mainProcessedWrapper.style.display = isTextMode ? 'flex' : 'none';
                    imageSearchContainer.style.display = isTextMode ? 'none' : 'block';
                    stageOptions.style.display = isTextMode ? 'flex' : 'none';
                } else {
                    button.classList.toggle('active');
                    const filterWrapper = stageCard.querySelector(`.filter-input-wrapper[data-filter-type="${type}"]`);
                    if (filterWrapper) {
                        filterWrapper.style.display = button.classList.contains('active') ? 'block' : 'none';
                        if (button.classList.contains('active')) { filterWrapper.querySelector('input').focus(); }
                    }
                }
                const genImageBtn = stageCard.querySelector('.type-btn[data-type="gen_image"]');
                if (genImageBtn && genImageBtn.classList.contains('active')) {
                    const textBtn = stageCard.querySelector('.type-btn[data-type="text"]');
                    if (!textBtn.classList.contains('active')) {
                        textBtn.click();
                    }
                }
            });
        });

        const handleFileSelect = async (file) => {
            if (!file || !file.type.startsWith('image/')) return;
            const previewImage = stageCard.querySelector('.image-preview');
            const uploadInstructions = stageCard.querySelector('.upload-instructions');
            const removeImageBtn = stageCard.querySelector('.remove-image-btn');
            previewImage.src = URL.createObjectURL(file);
            previewImage.style.display = 'block';
            uploadInstructions.style.display = 'none';
            removeImageBtn.style.display = 'none';
            const formData = new FormData();
            formData.append('image', file);
            try {
                const response = await fetch('http://localhost:18028/upload_image', { method: 'POST', body: formData });
                if (!response.ok) throw new Error(`Upload failed: ${response.statusText}`);
                const data = await response.json();
                stageCard.tempImageName = data.temp_image_name;
                removeImageBtn.style.display = 'flex';
            } catch (error) {
                console.error("Image upload error:", error);
                alert("Lỗi upload ảnh. Vui lòng thử lại.");
                previewImage.style.display = 'none';
                uploadInstructions.innerHTML = `<i class="fas fa-cloud-upload-alt"></i><p>Kéo & thả ảnh hoặc <strong>nhấn để chọn file</strong></p>`;
                uploadInstructions.style.display = 'block';
                removeImageBtn.style.display = 'none';
                delete stageCard.tempImageName;
            }
        };
        
        stageCard.querySelector('.remove-image-btn').addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); stageCard.querySelector('.stage-input-file').value = ''; const previewImage = stageCard.querySelector('.image-preview'); const uploadInstructions = stageCard.querySelector('.upload-instructions'); const removeImageBtn = stageCard.querySelector('.remove-image-btn'); previewImage.src = ''; previewImage.style.display = 'none'; uploadInstructions.innerHTML = `<i class="fas fa-cloud-upload-alt"></i><p>Kéo & thả ảnh hoặc <strong>nhấn để chọn file</strong></p>`; uploadInstructions.style.display = 'block'; removeImageBtn.style.display = 'none'; delete stageCard.tempImageName; });
        stageCard.querySelector('.stage-input-file').addEventListener('change', e => handleFileSelect(e.target.files[0]));
        imageUploadZone.addEventListener('dragover', e => { e.preventDefault(); imageUploadZone.classList.add('dragover'); });
        imageUploadZone.addEventListener('dragleave', e => { e.preventDefault(); imageUploadZone.classList.remove('dragover'); });
        imageUploadZone.addEventListener('drop', e => { e.preventDefault(); imageUploadZone.classList.remove('dragover'); handleFileSelect(e.dataTransfer.files[0]); });
        stageCard.querySelector('.delete-stage')?.addEventListener('click', () => { stageCard.remove(); renumberStages(); });
        stageCard.querySelectorAll('.option-btn').forEach(button => {
            if (button.dataset.option !== 'expand') {
                button.addEventListener('click', () => {
                    button.classList.toggle('active');
                });
            }
        });
        return stageCard;
    }

    async function handleSearch(imageFileFromClick = null) {
        const clientStartTime = performance.now();
        currentPage = 1;
        isLoadingMore = false;
        lastSearchPayload = null;

        loadingIndicator.style.display = 'block';
        resultsContainer.innerHTML = '';
        document.getElementById('timingInfoDisplay').style.display = 'none';
        currentResponse = {};
        pushedFrames.clear();

        const allStages = Array.from(stagesContainer.querySelectorAll('.stage-card'));

        allStages.forEach(stage => {
            const displayWrapper = stage.querySelector('.main-processed-query-wrapper');
            if (displayWrapper) {
                displayWrapper.style.display = 'none';
                displayWrapper.querySelector('.processed-query-display').textContent = '';
            }
        });
        
        try {
            const precomputationPromises = allStages.map(async (stage) => {
                const genImageBtn = stage.querySelector('.type-btn[data-type="gen_image"]');
            });

            await Promise.all(precomputationPromises);
        
            const objectFilters = getObjectFilterData();
            let response;
            let searchEndpoint;
            let requestBody;

            if (imageFileFromClick) {
                searchEndpoint = '/search';
                const searchData = { models: ["bge"], filters: objectFilters, image_search_text: "", page: 1, page_size: PAGE_SIZE };
                lastSearchPayload = searchData;
                const formData = new FormData();
                formData.append('query_image', imageFileFromClick, imageFileFromClick.name);
                formData.append('search_data', JSON.stringify(searchData));
                requestBody = formData;
            } else if (allStages.length === 1) {
                searchEndpoint = '/search';
                const firstStage = allStages[0];
                const useBgeCaption = firstStage.querySelector('.option-btn[data-option="bge_caption"]').classList.contains('active');
                // START: ADD is_only_meta_mode HERE
                const searchData = { filters: objectFilters, page: 1, page_size: PAGE_SIZE, use_bge_caption: useBgeCaption, is_only_meta_mode: onlyMetaBtn.classList.contains('active') };
                // END: ADD is_only_meta_mode HERE                
                
                let hasPrimaryQuery = false;
                if (firstStage.querySelector('.type-btn[data-type="gen_image"]')?.classList.contains('active')) {
                    const mainInput = firstStage.querySelector('.main-query-input');
                    searchData.query_text = mainInput.value.trim();
                    searchData.generated_image_name = firstStage.generatedImageName;
                    searchData.models = ["unite"];
                    if (searchData.query_text) hasPrimaryQuery = true;
                } else if (firstStage.querySelector('.type-btn[data-type="image"].active')) {
                    const imageTextInput = firstStage.querySelector('.image-search-text-input');
                    if (firstStage.tempImageName) {
                        searchData.query_image_name = firstStage.tempImageName;
                        searchData.image_search_text = imageTextInput.value.trim();
                        searchData.models = ["bge"];
                        hasPrimaryQuery = true;
                    }
                } else {
                    const mainInput = firstStage.querySelector('.main-query-input');
                    searchData.query_text = mainInput.value.trim();
                    searchData.models = Array.from(document.querySelectorAll('#modelDropdown input:checked')).map(cb => cb.value);
                    searchData.enhance = firstStage.querySelector('.option-btn[data-option="enhance"]').classList.contains('active');
                    if (searchData.query_text) hasPrimaryQuery = true;
                }

                if (firstStage.querySelector('.type-btn[data-type="ocr"].active')) searchData.ocr_query = firstStage.querySelector('.ocr-filter-input').value.trim();
                if (firstStage.querySelector('.type-btn[data-type="asr"].active')) searchData.asr_query = firstStage.querySelector('.asr-filter-input').value.trim();
                if (!hasPrimaryQuery && !searchData.ocr_query && !searchData.asr_query) throw new Error("Please provide a main query or a filter.");
                if (hasPrimaryQuery && searchData.models && searchData.models.length === 0) throw new Error("Please select at least one model.");
                
                lastSearchPayload = searchData;
                const formData = new FormData();
                formData.append('search_data', JSON.stringify(searchData));
                requestBody = formData;
            } else {
                searchEndpoint = '/temporal_search';
                const stagesData = allStages.map(stage => {
                    const mainInput = stage.querySelector('.main-query-input');
                    const stageDatum = {
                        query: mainInput.value.trim(),
                        query_image_name: stage.tempImageName || null,
                        generated_image_name: stage.generatedImageName || null,
                        enhance: stage.querySelector('.option-btn[data-option="enhance"]')?.classList.contains('active') || false,
                        expand: stage.querySelector('.option-btn[data-option="expand"]')?.classList.contains('active') || false,
                        use_bge_caption: stage.querySelector('.option-btn[data-option="bge_caption"]')?.classList.contains('active') || false,
                        ocr_query: stage.querySelector('.type-btn[data-type="ocr"].active') ? stage.querySelector('.ocr-filter-input').value.trim() : null,
                        asr_query: stage.querySelector('.type-btn[data-type="asr"].active') ? stage.querySelector('.asr-filter-input').value.trim() : null,
                    };
                    return stageDatum;
                });
                // START: ADD is_only_meta_mode HERE
                const payload = { stages: stagesData, models: Array.from(document.querySelectorAll('#modelDropdown input:checked')).map(cb => cb.value), filters: objectFilters, ambiguous: ambiguousBtn.classList.contains('active'), page: 1, page_size: PAGE_SIZE, is_only_meta_mode: onlyMetaBtn.classList.contains('active') };
                // END: ADD is_only_meta_mode HERE
                lastSearchPayload = payload;
                requestBody = JSON.stringify(payload);
            }
            
            // *** INTEGRATION POINT FOR HISTORY ***
            if (lastSearchPayload) {
                saveSearchToHistory(lastSearchPayload);
            }
            // ************************************

            const fetchOptions = { method: 'POST' };
            if (requestBody instanceof FormData) { fetchOptions.body = requestBody; } 
            else { fetchOptions.headers = { 'Content-Type': 'application/json' }; fetchOptions.body = requestBody; }
            
            response = await fetch(BACKEND_URL + searchEndpoint, fetchOptions);
            if (!response.ok) { const err = await response.json(); throw new Error(err.detail || 'Unknown error from server.'); }
            currentResponse = await response.json();

            if (currentResponse.processed_query) {
                const firstStage = allStages[0];
                if (firstStage) {
                    const displayWrapper = firstStage.querySelector('.main-processed-query-wrapper');
                    const displaySpan = displayWrapper.querySelector('.processed-query-display');
                    displaySpan.textContent = currentResponse.processed_query;
                    displayWrapper.style.display = 'flex';
                }
            }
            if (currentResponse.processed_queries && Array.isArray(currentResponse.processed_queries)) {
                allStages.forEach((stage, index) => {
                    const processedQueryText = currentResponse.processed_queries[index];
                    if (processedQueryText) {
                        const displayWrapper = stage.querySelector('.main-processed-query-wrapper');
                        if (displayWrapper) {
                            const displaySpan = displayWrapper.querySelector('.processed-query-display');
                            displaySpan.textContent = processedQueryText;
                            displayWrapper.style.display = 'flex';
                        }
                    }
                });
            }

            totalResults = currentResponse.total_results || 0;
            displayTimingInfo(currentResponse.timing_info, clientStartTime);
            displayResults(currentResponse.results, false);

        } catch (error) {
            console.error("[ERROR] Search failed:", error);
            resultsContainer.innerHTML = `<p style="color: #ef4444; font-size: 1.1rem; text-align: center;"><strong>Error:</strong> ${error.message}</p>`;
        } finally {
            loadingIndicator.style.display = 'none';
        }
    }
    
    let debounceTimer = null;
    function loadMoreResults() {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(async () => {
            const resultsCount = resultsContainer.querySelectorAll('.result-item, .sequence-result-container').length;
            if (isLoadingMore || !lastSearchPayload || (totalResults > 0 && resultsCount >= totalResults)) { return; }

            isLoadingMore = true;
            currentPage++;

            const loader = document.createElement('div');
            loader.id = 'moreLoader';
            loader.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading more...';
            loader.style.textAlign = 'center';
            loader.style.padding = '20px';
            loader.style.fontSize = '1.2rem';
            resultsContainer.appendChild(loader);

            try {
                let response;
                let searchEndpoint;
                let requestBody;
                lastSearchPayload.page = currentPage;

                if (lastSearchPayload.stages) {
                    searchEndpoint = '/temporal_search';
                    requestBody = JSON.stringify(lastSearchPayload);
                } else {
                    searchEndpoint = '/search';
                    const formData = new FormData();
                    formData.append('search_data', JSON.stringify(lastSearchPayload));
                    requestBody = formData;
                }

                const fetchOptions = { method: 'POST' };
                if (requestBody instanceof FormData) { fetchOptions.body = requestBody; } 
                else { fetchOptions.headers = { 'Content-Type': 'application/json' }; fetchOptions.body = requestBody; }

                response = await fetch(BACKEND_URL + searchEndpoint, fetchOptions);
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to load more results.');
                }
                const data = await response.json();
                displayResults(data.results, true);
            } catch (error) {
                console.error("Error loading more results:", error);
                loader.textContent = `Error: ${error.message}`;
            } finally {
                isLoadingMore = false;
                const existingLoader = document.getElementById('moreLoader');
                if (existingLoader) existingLoader.remove();
                debounceTimer = null;
            }
        }, 300);
    }
    
    function getObjectFilterData(){const e={};if(enableCountFilter.checked){const t={};countFilterControls.querySelectorAll(".count-filter-row").forEach(e=>{const o=e.querySelector(".count-checkbox");if(o.checked){const r=e.querySelector(".condition-input").value.trim();if(r){let s=e.classList.contains("custom")?e.querySelector(".custom-object-name").value.trim().toLowerCase():o.dataset.object;s&&(t[s]=r)}}}),Object.keys(t).length>0&&(e.counting={conditions:t})}if(enablePositionFilter.checked&&drawnBoxes.length>0){const o=drawnBoxes.filter(e=>e.label).map(e=>({label:e.label,box:[e.x/posCanvas.width,e.y/posCanvas.height,(e.x+e.w)/posCanvas.width,(e.y+e.h)/posCanvas.height]}));o.length>0&&(e.positioning={boxes:o})}return objectFilterBtn.classList.toggle("active",enableCountFilter.checked||enablePositionFilter.checked),(enableCountFilter.checked||enablePositionFilter.checked)&&Object.keys(e).length>0?e:null}
    function createCountRow(e = "", t = !1) { const o = document.createElement("div"); o.className = "count-filter-row" + (t ? " custom" : ""); const r = `<input type="checkbox" class="count-checkbox" data-object="${e || "custom"}">`, s = t ? `<input type="text" class="filter-input custom-object-name" placeholder="object name">` : `<label>${e}</label>`, n = `<input type="text" class="filter-input condition-input" placeholder="e.g., >=1">`, i = t ? `<button class="remove-custom-btn">&times;</button>` : ""; o.innerHTML = `${r}${s}${n}${i}`; const a = o.querySelector(".count-checkbox"); const l = () => o.classList.toggle("active-row", a.checked); o.addEventListener("click", e => { e.target.tagName !== "INPUT" && e.target.tagName !== "BUTTON" && (a.checked = !a.checked, l()) }); a.addEventListener("change", l); const conditionInput = o.querySelector('.condition-input'); conditionInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); if (conditionInput.value.trim() !== '') { a.checked = true; l(); } conditionInput.blur(); } }); t && o.querySelector(".remove-custom-btn").addEventListener("click", () => o.remove()); l(); return o; }
    PREDEFINED_OBJECTS.forEach(e=>countFilterControls.appendChild(createCountRow(e))),addCustomCountBtn.addEventListener("click",()=>countFilterControls.appendChild(createCountRow("",!0)));
    function redrawCanvas(){if(!posCtx)return;posCtx.clearRect(0,0,posCanvas.width,posCanvas.height),posCtx.strokeStyle="#FFD700",posCtx.lineWidth=2,drawnBoxes.forEach((e,t)=>{posCtx.strokeRect(e.x,e.y,e.w,e.h),posCtx.font="14px 'Poppins'",posCtx.fillStyle="#FFD700",posCtx.fillText(`${t+1}: ${e.label||"no label"}`,e.x+5,e.y+16)}),isDrawing&&(posCtx.strokeStyle="var(--accent-pink)",posCtx.strokeRect(startX,startY,currentX-startX,currentY-startY))}
    function updateDrawnBoxesList(){drawnBoxesList.innerHTML="",drawnBoxes.forEach((e,t)=>{const o=document.createElement("div");o.className="drawn-box-item",o.textContent=`Box ${t+1}: ${e.label||"(unlabeled)"}`,drawnBoxesList.appendChild(o)})}posCanvas.addEventListener("pointerdown",e=>{isDrawing=!0,startX=e.offsetX,startY=e.offsetY}),posCanvas.addEventListener("pointermove",e=>{isDrawing&&(currentX=e.offsetX,currentY=e.offsetY,redrawCanvas())}),posCanvas.addEventListener("pointerup",e=>{if(!isDrawing)return;isDrawing=!1;const t=e.offsetX,o=e.offsetY,r={x:Math.min(startX,t),y:Math.min(startY,o),w:Math.abs(t-startX),h:Math.abs(o-startY),label:""};r.w>5&&r.h>5&&drawnBoxes.push(r),redrawCanvas(),updateDrawnBoxesList()});
    function renumberStages() { stagesContainer.querySelectorAll('.stage-card').forEach((card, index) => { const stageNumberEl = card.querySelector('.stage-number'); if (stageNumberEl) stageNumberEl.textContent = index + 1; const deleteBtn = card.querySelector('.delete-stage'); if (deleteBtn) { deleteBtn.style.display = stagesContainer.children.length > 1 ? 'flex' : 'none'; } }); }
    function addStageToEnd() { const newStage = createStageCard(0); stagesContainer.appendChild(newStage); renumberStages(); focusOnStageInput(newStage); }
    function addStageToStart() { const newStage = createStageCard(0); stagesContainer.insertAdjacentElement('afterbegin', newStage); renumberStages(); focusOnStageInput(newStage); }
    function removeStageFromEnd() { if (stagesContainer.children.length > 1) { stagesContainer.lastChild.remove(); renumberStages(); focusOnStageInput(stagesContainer.lastChild); } }
    function removeStageFromStart() { if (stagesContainer.children.length > 1) { stagesContainer.firstChild.remove(); renumberStages(); focusOnStageInput(stagesContainer.firstChild); } }
    
    // ===============================================
    // START: SEARCH HISTORY FUNCTIONS
    // ===============================================

    /**
     * Lấy danh sách lịch sử từ localStorage và parse nó.
     * @returns {Array} Mảng các mục lịch sử.
     */
    function getSearchHistory() {
        try {
            const historyJSON = localStorage.getItem('searchHistory');
            return historyJSON ? JSON.parse(historyJSON) : [];
        } catch (e) {
            console.error("Could not read search history from localStorage", e);
            return [];
        }
    }
    /**
     * Lưu một payload tìm kiếm vào lịch sử.
     * @param {object} searchPayload - Đối tượng chứa thông tin tìm kiếm.
     */
    function saveSearchToHistory(searchPayload) {
        if (!searchPayload) return;

        // Create a user-friendly name for the history item
        let name = "Untitled Search";

        if (searchPayload.stages) { // This is a temporal (multi-stage) search
            // THIS IS THE CORRECTED LOGIC
            name = searchPayload.stages.map((s, index) => {
                let stageDesc = "Filter-Only Stage"; // Default description
                if (s.query) {
                    stageDesc = s.query;
                } else if (s.query_image_name) {
                    stageDesc = "Image Query";
                } else if (s.ocr_query || s.asr_query) {
                    stageDesc = `Filters: [${[s.ocr_query, s.asr_query].filter(Boolean).join(', ')}]`;
                }
                return `${index + 1}. ${stageDesc}`;
            }).join('\n'); // Join with a newline character for multi-line storage
        } else { // This is a single-stage search
            name = searchPayload.query_text || searchPayload.image_search_text || "Filter-Only Search";
        }

        if (name.trim() === "") name = "Filter-Only Search";

        const history = getSearchHistory();
        const newEntry = {
            id: Date.now(),
            name: name,
            timestamp: new Date().toLocaleString(),
            payload: searchPayload
        };

        // Add the new entry to the top and trim the list if it's too long
        const updatedHistory = [newEntry, ...history].slice(0, MAX_HISTORY_ITEMS);

        try {
            localStorage.setItem('searchHistory', JSON.stringify(updatedHistory));
        } catch (e) {
            console.error("Could not save search history to localStorage. It might be full.", e);
        }
    }

    /**
     * Render danh sách lịch sử ra modal.
     */
    function renderHistoryList() {
        const history = getSearchHistory();
        historyListContainer.innerHTML = ''; // Clear the old list

        if (history.length === 0) {
            historyListContainer.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 20px;">Your search history is empty.</p>';
            return;
        }

        history.forEach(item => {
            const itemEl = document.createElement('div');
            itemEl.className = 'history-item';
            itemEl.dataset.historyId = item.id;
            
            // Replace stored newlines (\n) with HTML line breaks (<br>) for correct display
            const formattedName = item.name.replace(/\n/g, '<br>');

            itemEl.innerHTML = `
                <div class="history-item-content">
                    <div class="history-item-query">${formattedName}</div>
                    <div class="history-item-details">${item.timestamp}</div>
                </div>
                <div class="history-item-actions">
                    <button class="delete-history-item" title="Delete this entry"><i class="fas fa-trash-alt"></i></button>
                </div>
            `;

            // Event listener to reload the search when clicked
            itemEl.querySelector('.history-item-content').addEventListener('click', () => {
                loadSearchFromHistory(item.id);
                historyModal.style.display = 'none';
            });

            // Event listener for the delete button
            itemEl.querySelector('.delete-history-item').addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent the reload event from firing
                deleteHistoryItem(item.id);
            });

            historyListContainer.appendChild(itemEl);
        });
    }
    
    /**
     * Xóa một mục cụ thể khỏi lịch sử.
     * @param {number} id - ID của mục cần xóa.
     */
    function deleteHistoryItem(id) {
        let history = getSearchHistory();
        const updatedHistory = history.filter(item => item.id !== id);
        localStorage.setItem('searchHistory', JSON.stringify(updatedHistory));
        renderHistoryList(); // Cập nhật lại giao diện ngay lập tức
    }
    
    /**
     * Tải lại trạng thái tìm kiếm từ một mục lịch sử.
     * @param {number} id - ID của mục cần tải.
     */
    function loadSearchFromHistory(id) {
        const history = getSearchHistory();
        const itemToLoad = history.find(h => h.id === id);
        if (!itemToLoad) {
            showToast("Could not find that history item.", 3000, 'error');
            return;
        }
        
        const payload = itemToLoad.payload;
        
        // Reset UI before loading
        stagesContainer.innerHTML = '';
        
        // This handles both single-stage (becomes an array of 1) and multi-stage payloads
        const stagesData = payload.stages || [payload]; 
        
        stagesData.forEach((stageData, index) => {
            const stageCard = createStageCard(index + 1);
            
            // Set query text/image
            if (stageData.query_image_name) {
                stageCard.querySelector('.type-btn[data-type="image"]').click();
                const imageSearchText = stageCard.querySelector('.image-search-text-input');
                imageSearchText.value = stageData.image_search_text || '';
                const uploadInstructions = stageCard.querySelector('.upload-instructions p');
                uploadInstructions.innerHTML = `<strong>Please re-upload image for this search.</strong>`;
            } else {
                // Correctly handle 'query' (from multi-stage) and 'query_text' (from single-stage)
                stageCard.querySelector('.main-query-input').value = stageData.query || stageData.query_text || '';
            }
            
            // Set OCR/ASR filters
            if (stageData.ocr_query) {
                stageCard.querySelector('.type-btn[data-type="ocr"]').click();
                stageCard.querySelector('.ocr-filter-input').value = stageData.ocr_query;
            }
            if (stageData.asr_query) {
                stageCard.querySelector('.type-btn[data-type="asr"]').click();
                stageCard.querySelector('.asr-filter-input').value = stageData.asr_query;
            }
            
            // Set options (enhance, caption)
            if (stageData.enhance) stageCard.querySelector('.option-btn[data-option="enhance"]').classList.add('active');
            if (stageData.use_bge_caption) stageCard.querySelector('.option-btn[data-option="bge_caption"]').classList.add('active');
            
            stagesContainer.appendChild(stageCard);
        });
        renumberStages();
        
        // Restore global options
        document.querySelectorAll('#modelDropdown input[type="checkbox"]').forEach(cb => {
            cb.checked = (payload.models || ['beit3', 'bge', 'ops_mm']).includes(cb.value);
        });
        
        clusterBtn.classList.toggle('active', payload.cluster === true);
        ambiguousBtn.classList.toggle('active', payload.ambiguous === true);
        
        showToast("Search loaded from history. Press Search to run.", 3000, 'info');
    }
    // ===============================================
    // END: SEARCH HISTORY FUNCTIONS
    // ===============================================


    setupImageObserver();
    setupUser();
    initializeDresState();
    function applyTheme(themeName) {
        document.body.className = themeName === 'default' ? '' : themeName;
        localStorage.setItem('videoSearchTheme', themeName);
        const themeLabel = THEMES[themeName] || 'Default Dark';
        themeSwitcherBtn.querySelector('span').innerHTML = `<i class="fas fa-palette"></i> ${themeLabel}`;
    }

    function initTheme() {
        const fragment = document.createDocumentFragment();
        for (const [className, name] of Object.entries(THEMES)) {
            const item = document.createElement('div');
            item.className = 'theme-dropdown-item';
            item.dataset.theme = className;
            const swatch = document.createElement('div');
            swatch.className = 'theme-color-swatch';
            const dummy = document.createElement('div');
            dummy.style.display = 'none';
            document.body.appendChild(dummy);
            const originalClass = document.body.className;
            document.body.className = className === 'default' ? '' : className;
            const primaryGradient = getComputedStyle(dummy).getPropertyValue('--primary-gradient');
            swatch.style.background = primaryGradient;
            document.body.className = originalClass;
            dummy.remove();
            item.appendChild(swatch);
            item.append(name);
            item.addEventListener('click', () => {
                applyTheme(className);
                themeDropdown.style.display = 'none';
            });
            fragment.appendChild(item);
        }
        themeDropdown.appendChild(fragment);
        const savedTheme = localStorage.getItem('videoSearchTheme') || 'default';
        applyTheme(savedTheme);
    }

    initTheme();
    function closeSidebar() {
        nearbyFramesSidebar.style.display = 'none';
        activeTrakeItemForSidebar = null;
    }
    sidebarCloseBtn.addEventListener('click', closeSidebar);
    sidebarOverlay.addEventListener('click', closeSidebar);

    themeSwitcherBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        themeDropdown.style.display = themeDropdown.style.display === 'block' ? 'none' : 'block';
    });

    trakeBtn.addEventListener('click', () => {
        const isActive = trakeBtn.classList.toggle('active');
        trakePanelContainer.style.display = isActive ? 'block' : 'none';
    });
    // START: ADD THIS ENTIRE BLOCK
    onlyMetaBtn.addEventListener('click', () => {
        const isActive = onlyMetaBtn.classList.toggle('active');
        const modelCheckboxes = document.querySelectorAll('#modelDropdown input[type="checkbox"]');
        const metaClipCheckbox = document.getElementById('model-metaclip2');

        if (isActive) {
            // "OnlyMeta" mode is ON
            modelCheckboxes.forEach(cb => {
                cb.checked = false;
            });
            if (metaClipCheckbox) {
                metaClipCheckbox.checked = true;
            }
        } else {
            // "OnlyMeta" mode is OFF, restore all models
            modelCheckboxes.forEach(cb => {
                cb.checked = true;
            });
        }
    });
    // END: ADD THIS ENTIRE BLOCK

    function pushToTrakePanel(shotData) {
        if (!ws || ws.readyState !== WebSocket.OPEN) { showToast("Real-time connection not active.", 3000, 'error'); return; }
        if (!shotData || !shotData.filepath) return;
        if (trakeGrid.querySelector(`.result-item[data-filepath="${shotData.filepath}"]`)) { showToast("Frame is already in the Trake Panel.", 2000, 'warning'); return; }
        ws.send(JSON.stringify({ type: 'trake_add', data: { shot: shotData } }));
        if (!trakeBtn.classList.contains('active')) { trakeBtn.click(); }
    }

    pushToTrakeBtn.addEventListener('click', () => {
        if (videoPreviewModal.style.display === 'flex' && currentVideoPreviewData) {
            const currentTime = videoPlayer.currentTime;
            const frameId = Math.round(currentTime * currentVideoPreviewData.fps);
            const canvas = document.createElement('canvas');
            canvas.width = videoPlayer.videoWidth;
            canvas.height = videoPlayer.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoPlayer, 0, 0, canvas.width, canvas.height);
            const thumbnailUrl = canvas.toDataURL('image/jpeg', 0.8);
            const shotData = { video_id: currentVideoPreviewData.videoId, frame_id: frameId, fps: currentVideoPreviewData.fps, filepath: `dynamic-frame-${currentVideoPreviewData.videoId}-${frameId}`, url: thumbnailUrl, is_dynamic: true };
            pushToTrakePanel(shotData);
        }
    });

    let draggedItem = null;
    trakeGrid.addEventListener('dragstart', e => {
        draggedItem = e.target.closest('.result-item');
        setTimeout(() => { if(draggedItem) draggedItem.classList.add('ghost'); }, 0);
    });
    trakeGrid.addEventListener('dragend', e => {
        if(draggedItem) {
            draggedItem.classList.remove('ghost');
            draggedItem = null;
            const newOrder = [...trakeGrid.querySelectorAll('.result-item')].map(item => item.dataset.filepath);
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'trake_reorder', data: { order: newOrder } }));
            }
        }
    });
    trakeGrid.addEventListener('dragover', e => {
        e.preventDefault();
        const afterElement = getDragAfterElement(trakeGrid, e.clientX);
        const dragging = document.querySelector('.ghost');
        if (dragging) {
            if (afterElement == null) { trakeGrid.appendChild(dragging); } 
            else { trakeGrid.insertBefore(dragging, afterElement); }
        }
    });

    function getDragAfterElement(container, x) {
        const draggableElements = [...container.querySelectorAll('.result-item:not(.ghost)')];
        return draggableElements.reduce((closest, child) => {
            const box = child.getBoundingClientRect();
            const offset = x - box.left - box.width / 2;
            if (offset < 0 && offset > closest.offset) {
                return { offset: offset, element: child };
            } else {
                return closest;
            }
        }, { offset: Number.NEGATIVE_INFINITY }).element;
    }
    function handleNearbyFramesClick(event) {
        event.stopPropagation();
        const itemElement = event.target.closest('.result-item');
        if (itemElement && itemElement.shotData) {
            toggleNearbyFrames(itemElement, itemElement.shotData);
        } else {
            console.error("Could not find shot data for nearby frames button.", itemElement);
            showToast("Error: Data for this frame is missing.", 3000, 'error');
        }
    }

    async function toggleNearbyFrames(trakeItemElement, shotData) {
        activeTrakeItemForSidebar = trakeItemElement;
        sidebarGrid.innerHTML = '<div class="sidebar-item loading" style="justify-content: center;"><i class="fas fa-spinner fa-spin"></i></div>';
        sidebarTitle.textContent = `Nearby: V${shotData.video_id}, F${shotData.frame_id}`;
        nearbyFramesSidebar.style.display = 'flex';
        let currentShotData = { ...shotData };

        if (!currentShotData.fps) {
            try {
                if (!currentShotData.video_id) throw new Error("Missing video_id");
                const infoResponse = await fetch(`http://localhost:18028/video_info/${currentShotData.video_id}`);
                if (!infoResponse.ok) throw new Error(`Server error ${infoResponse.status}`);
                const videoInfo = await infoResponse.json();
                currentShotData.fps = videoInfo.fps;
                trakeItemElement.shotData.fps = videoInfo.fps;
            } catch (error) {
                sidebarGrid.innerHTML = `<div style="color: #ef4444;">Error: ${error.message}</div>`;
                return;
            }
        }
        
        try {
            const originalTime = currentShotData.frame_id / currentShotData.fps;
            const timeOffsets = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0];
            const framePromises = timeOffsets.map(offset => fetchFrameAtTime(currentShotData.video_id, originalTime + offset));
            const frameResults = await Promise.all(framePromises);
            
            sidebarGrid.innerHTML = '';
            frameResults.forEach((result, index) => {
                if (result.success) {
                    const offset = timeOffsets[index];
                    const newItem = document.createElement('div');
                    newItem.className = 'sidebar-item';
                    newItem.innerHTML = `<img src="${result.imageData}" /><span class="time-offset">${offset === 0 ? 'Original' : (offset > 0 ? `+${offset.toFixed(1)}s` : `${offset.toFixed(1)}s`)}</span>`;
                    
                    newItem.addEventListener('click', () => {
                        const newFrameId = Math.round((originalTime + offset) * currentShotData.fps);
                        const newShotData = { ...currentShotData, frame_id: newFrameId, url: result.imageData };
                        
                        if (activeTrakeItemForSidebar && ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({
                                type: 'trake_replace',
                                data: { filepath: activeTrakeItemForSidebar.dataset.filepath, newShot: newShotData }
                            }));
                        }
                        closeSidebar();
                    });
                    sidebarGrid.appendChild(newItem);
                }
            });
        } catch (error) {
            sidebarGrid.innerHTML = `<div style="color: #ef4444;">Error loading frames: ${error.message}</div>`;
        }
    }

    async function fetchFrameAtTime(videoId, timestamp) {
        try {
            const formData = new FormData();
            formData.append('video_id', videoId);
            formData.append('timestamp', timestamp);
            const response = await fetch('http://localhost:18028/get_frame_at_timestamp', { method: 'POST', body: formData });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Server error');
            }
            const data = await response.json();
            return { success: true, imageData: data.image_data };
        } catch (error) {
            console.error(`Failed to fetch frame for ${videoId} at ${timestamp}s:`, error);
            return { success: false, error: error.message };
        }
    }

    trakePanelContainer.addEventListener('mouseenter', () => { isMouseOverTrakePanel = true; });
    trakePanelContainer.addEventListener('mouseleave', () => { isMouseOverTrakePanel = false; });
    
    async function submitAllTrakeFrames() {
        if (!isMouseOverTrakePanel) return;
        const itemsToSubmit = Array.from(trakeGrid.querySelectorAll('.result-item'));
        if (itemsToSubmit.length === 0) {
            showToast("Trake Panel is empty.", 2000, 'info');
            return;
        }
        if (!confirm(`Submit all ${itemsToSubmit.length} frames in the Trake Panel to DRES?`)) { return; }
        showToast(`Submitting ${itemsToSubmit.length} frames...`, 5000, 'info');
        const submissionPromises = itemsToSubmit.map(item => {
            const shotData = { video_id: item.dataset.videoId, frame_id: parseInt(item.dataset.frameId, 10), filepath: item.dataset.filepath, url: item.querySelector('img').src };
            return handleSubmitToDRES(shotData, true);
        });
        await Promise.all(submissionPromises);
        showToast("All submissions from Trake Panel processed.", 3000, 'success');
    }
    const initialStage = createStageCard(1);
    stagesContainer.appendChild(initialStage);
    focusOnStageInput(initialStage);

    addStageBtn.addEventListener('click', addStageToEnd);
    removeStageBtn.addEventListener('click', removeStageFromEnd);
    searchBtn.addEventListener('click', () => handleSearch(null));
    document.getElementById('resetBtn').addEventListener('click', () => {
        if (confirm('Are you sure you want to reset the search panel?')) {
            stagesContainer.innerHTML = '';
            addStageToStart();
            currentResponse = {};
            resultsContainer.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding-top: 100px; font-size: 1.5rem;">Sử dụng bảng điều khiển bên trái để bắt đầu tìm kiếm.</p>';
            document.getElementById('timingInfoDisplay').style.display = 'none';
            teamworkGrid.innerHTML = '';
            pushedFrames.clear();
            submittedFrames.clear();
            document.getElementById('correctSubmissionPanel').style.display = 'none';
            googleSearchInput.value = '';
            googleResultsContainer.innerHTML = '';
            googleResultsWrapper.style.display = 'none';
        }
    });            
    clusterBtn.addEventListener('click', () => {
        clusterBtn.classList.toggle('active');
        if (currentResponse.results && !isLoadingMore) {
            currentPage = 1; 
            const allData = currentResponse.results;
            displayResults(allData, false);
        }
    });
    ambiguousBtn.addEventListener('click', () => { ambiguousBtn.classList.toggle('active'); });
    modelSelectBtn.addEventListener('click', (event) => { event.stopPropagation(); const isOpening = modelDropdown.style.display !== 'block'; modelDropdown.style.display = isOpening ? 'block' : 'none'; if (isOpening) { focusedModelIndex = 0; updateModelFocus(); } else { clearModelFocus(); } });
    window.addEventListener('click', (e) => { if (!modelDropdown.contains(e.target) && !modelSelectBtn.contains(e.target)) { if (modelDropdown.style.display === 'block') { modelDropdown.style.display = 'none'; clearModelFocus(); } } });
    closeImageModalBtn.addEventListener('click', () => { imageModal.style.display = "none"; });
    imageModal.addEventListener('click', (e) => { if (e.target === imageModal) { imageModal.style.display = "none"; } });
    closeTemporalModalBtn.addEventListener('click', () => { temporalContextModal.style.display = "none"; });
    temporalContextModal.addEventListener('click', (e) => { if (e.target === temporalContextModal) { temporalContextModal.style.display = "none"; } });
    
    function closeVideoModal() {
        document.body.classList.remove('video-modal-active');
        videoPreviewModal.style.display = "none";
        videoPlayer.pause();
        videoPlayer.src = "";
        currentVideoPreviewData = null;
    
        // --- TIMELINE CLEANUP ---
        if (animationFrameId_timeline !== null) {
            cancelAnimationFrame(animationFrameId_timeline);
            animationFrameId_timeline = null;
        }
        if (timelineObserver) {
            timelineObserver.disconnect();
        }
        thumbnailQueue.length = 0;
        isGenerating = false;
        isTempVideoReady = false;
        tempVideo.src = "";
        thumbnailCache.clear();
        videoThumbnailsStrip.innerHTML = "";
    }

    closeVideoModalBtn.addEventListener('click', closeVideoModal);
    videoPreviewModal.addEventListener('click', (e) => { if (e.target === videoPreviewModal) { closeVideoModal(); } });
    modalCloseBtn.addEventListener('click', () => objectFilterModal.style.display = 'none');

    function resetDresState() {
        dresSessionId = null; dresEvaluationId = null;
        sessionStorage.removeItem('dresSessionId'); sessionStorage.removeItem('dresEvaluationId');
        dresStatus.textContent = 'Status: Not logged in.'; dresStatus.style.color = 'var(--text-secondary)';
        dresEvaluationSelect.innerHTML = ''; dresEvaluationSelect.disabled = true;
    }

    async function revalidateAndFetchEvaluations() {
        if (!dresSessionId) return;
        dresStatus.textContent = 'Validating session & fetching evaluations...';
        dresStatus.style.color = 'var(--text-secondary)';
        try {
            const evalResponse = await fetch(`http://localhost:18028/dres/list_evaluations?session=${dresSessionId}`);
            if (!evalResponse.ok) { throw new Error('Session invalid or expired. Please log in again.'); }
            const evaluations = await evalResponse.json();
            dresEvaluationSelect.innerHTML = '';
            evaluations.forEach(ev => {
                if (ev.status === 'ACTIVE') {
                    const option = document.createElement('option');
                    option.value = ev.id;
                    option.textContent = ev.name;
                    dresEvaluationSelect.appendChild(option);
                }
            });
            if (dresEvaluationSelect.options.length > 0) {
                const storedEvalId = sessionStorage.getItem('dresEvaluationId');
                if (storedEvalId && dresEvaluationSelect.querySelector(`option[value="${storedEvalId}"]`)) { dresEvaluationSelect.value = storedEvalId; }
                dresEvaluationId = dresEvaluationSelect.value;
                sessionStorage.setItem('dresEvaluationId', dresEvaluationId);
                dresStatus.textContent = `Ready to submit to: ${dresEvaluationSelect.options[dresEvaluationSelect.selectedIndex].text}`;
                dresStatus.style.color = 'var(--accent-blue)';
            } else {
                dresStatus.textContent = 'Logged in, but no active evaluations found.';
                dresStatus.style.color = 'var(--text-secondary)';
            }
            dresEvaluationSelect.disabled = false;
        } catch (error) {
            console.error("DRES Re-validation Error:", error.message);
            showToast(error.message, 4000, 'error');
            resetDresState();
        }
    }
    
    function initializeDresState() {
        dresSessionId = sessionStorage.getItem('dresSessionId');
        if (dresSessionId) { revalidateAndFetchEvaluations(); }
    }
    
    dresModalCloseBtn.addEventListener('click', () => dresModal.style.display = 'none');
    dresBtn.addEventListener('click', () => {
        if (dresSessionId) {
            dresInitialView.style.display = 'none'; dresLoginView.style.display = 'none'; dresEvaluationView.style.display = 'block';
            const selectedOption = dresEvaluationSelect.options[dresEvaluationSelect.selectedIndex];
            dresStatus.textContent = selectedOption ? `Ready to submit to: ${selectedOption.text}` : 'Logged in, please select an evaluation.';
            dresStatus.style.color = 'var(--accent-blue)';
        } else {
            dresInitialView.style.display = 'block'; dresLoginView.style.display = 'none'; dresEvaluationView.style.display = 'none';
            dresStatus.textContent = 'Status: Not logged in.'; dresStatus.style.color = 'var(--text-secondary)';
        }
        dresModal.style.display = 'flex';
    });
    dresShowLoginBtn.addEventListener('click', () => { dresInitialView.style.display = 'none'; dresLoginView.style.display = 'flex'; dresUsername.focus(); });
    dresLoginBtn.addEventListener('click', async () => {
        const user = dresUsername.value; const pass = dresPassword.value;
        if (!user || !pass) { alert('Please enter username and password.'); return; }
        dresStatus.textContent = 'Logging in...'; dresStatus.style.color = 'var(--text-secondary)';
        try {
            const response = await fetch('http://localhost:18028/dres/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: user, password: pass })
            });
            if (!response.ok) { const err = await response.json(); throw new Error(err.detail || 'Login failed'); }
            const data = await response.json();
            dresSessionId = data.sessionId;
            sessionStorage.setItem('dresSessionId', dresSessionId);
            await revalidateAndFetchEvaluations();
            showToast('DRES Login successful!', 3000, 'success');
            dresModal.style.display = 'none';
        } catch (error) {
            dresStatus.textContent = `Error: ${error.message}`; dresStatus.style.color = '#ef4444'; resetDresState();
        }
    });
    dresEvaluationSelect.addEventListener('change', () => { dresEvaluationId = dresEvaluationSelect.value; sessionStorage.setItem('dresEvaluationId', dresEvaluationId); dresStatus.textContent = `Ready to submit to: ${dresEvaluationSelect.options[dresEvaluationSelect.selectedIndex].text}`; });
    objectFilterBtn.addEventListener('click', () => { objectFilterModal.style.display = 'flex'; });

    // *** HISTORY EVENT LISTENERS ***
    historyBtn.addEventListener('click', () => {
        renderHistoryList(); // Luôn render lại để lấy dữ liệu mới nhất
        historyModal.style.display = 'flex';
    });
    historyModalCloseBtn.addEventListener('click', () => {
        historyModal.style.display = 'none';
    });
    historyModal.addEventListener('click', (e) => {
        if (e.target === historyModal) {
            historyModal.style.display = 'none';
        }
    });
    clearHistoryBtn.addEventListener('click', () => {
        if (confirm("Are you sure you want to clear ALL search history? This cannot be undone.")) {
            localStorage.removeItem('searchHistory');
            renderHistoryList(); // Cập nhật lại UI trống
        }
    });
    // *******************************


    window.addEventListener('keydown', (event) => {
        const activeElement = document.activeElement;
        const isTyping = activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA');

        if (videoPreviewModal.style.display === 'flex' && event.code === 'Space' && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            if (event.shiftKey) { submitCurrentFrameBtn.click(); } 
            else { pushCurrentFrameBtn.click(); }
            return;
        }
        if (event.key === 'Escape') {
            event.preventDefault();
            if (isTyping) { activeElement.blur(); return; }
            if (videoPreviewModal.style.display === 'flex') { closeVideoModal(); return; }
            if (imageModal.style.display === 'flex') { imageModal.style.display = 'none'; return; }
            if (temporalContextModal.style.display === 'flex') { temporalContextModal.style.display = 'none'; return; }
            if (objectFilterModal.style.display === 'flex') { objectFilterModal.style.display = 'none'; return; }
            if (dresModal.style.display === 'flex') { dresModal.style.display = 'none'; return; }
            if (historyModal.style.display === 'flex') { historyModal.style.display = 'none'; return; } // Close history modal
            if (modelDropdown.style.display === 'block') { modelDropdown.style.display = 'none'; clearModelFocus(); return; }
        }
        
        const isModalVisible = Array.from(document.querySelectorAll('.modal-overlay')).some(m => m.style.display === 'flex' || m.style.display === 'block');
        
        if ((event.ctrlKey || event.metaKey)) {
            if (event.key.toLowerCase() === 'h' && !event.shiftKey && !event.altKey && !isTyping) { // History shortcut
                event.preventDefault();
                historyBtn.click();
                return;
            }
            if (event.key.toLowerCase() === 'f' && !event.shiftKey && !event.altKey) {
                event.preventDefault();
                if (objectFilterModal.style.display === 'flex') { objectFilterModal.style.display = 'none'; } 
                else { objectFilterModal.style.display = 'flex'; }
                return;
            }
            if (event.key.toLowerCase() === 'f' && event.shiftKey && !event.altKey) {
                event.preventDefault();
                document.getElementById('enableCountFilter').click();
                document.getElementById('enablePositionFilter').click();
                objectFilterBtn.classList.toggle('active', document.getElementById('enableCountFilter').checked || document.getElementById('enablePositionFilter').checked);
                return;
            }
            if (event.key.toLowerCase() === 'm' && !event.shiftKey && !event.altKey) {
                event.preventDefault(); modelSelectBtn.click(); return;
            }
        }

        if (event.code === 'Space' && (event.ctrlKey || event.metaKey)) {
            if (!isModalVisible && currentlyHoveredItemData && currentlyHoveredItemElement) {
                event.preventDefault();
                const itemData = currentlyHoveredItemData;
                const itemElement = currentlyHoveredItemElement;
                const source = itemElement.dataset.source;
                if (event.shiftKey) { // This means submit
                    if (itemData.external_url) {
                        handleGoogleImageAction(itemData.external_url, 'submit', itemElement);
                    } else {
                        // TẠO ĐỐI TƯỢNG SUBMISSION HOÀN CHỈNH
                        const submissionData = {
                            ...itemData, // Sao chép tất cả các thuộc tính cũ (video_id, frame_id, ...)
                            url: `./webp_keyframes/${itemData.frame_name}` // Bổ sung thuộc tính url còn thiếu
                        };
                        handleSubmitToDRES(submissionData, true);
                    }
                }
                else {
                    if (source === 'teamwork') {
                        if (!ws || ws.readyState !== WebSocket.OPEN) { alert("Teamwork connection is not available."); return; }
                        ws.send(JSON.stringify({ type: 'remove_frame', data: { filepath: itemData.filepath, user: { name: username } } }));
                    } else {
                        if (itemData.external_url) { handleGoogleImageAction(itemData.external_url, 'push', itemElement); } 
                        else { pushToTeamworkPanel(itemData); }
                    }
                }
                return;
            }
            else if (isMouseOverTrakePanel && event.shiftKey) {
                event.preventDefault(); submitAllTrakeFrames();
            }
        }

        if (modelDropdown.style.display === 'block') {
            event.preventDefault();
            const items = modelDropdown.querySelectorAll('.model-dropdown-item');
            if (items.length > 0) {
                switch (event.key) {
                    case 'ArrowDown': focusedModelIndex = (focusedModelIndex + 1) % items.length; updateModelFocus(); break;
                    case 'ArrowUp': focusedModelIndex = (focusedModelIndex - 1 + items.length) % items.length; updateModelFocus(); break;
                    case 'Enter': case ' ': if (focusedModelIndex > -1) { items[focusedModelIndex].querySelector('input[type="checkbox"]')?.click(); } break;
                }
            }
            return;
        }

        if (event.key === 'Enter') {
            if (isTyping && event.shiftKey) { return; }
            if (dresModal.style.display === 'flex') {
                event.preventDefault();
                if (dresLoginView.style.display === 'flex') { dresLoginBtn.click(); } 
                else if (dresInitialView.style.display === 'block') { dresShowLoginBtn.click(); }
                return;
            }
            if (usernameModal.style.display === 'flex') { event.preventDefault(); usernameSubmitBtn.click(); return; }
            event.preventDefault(); searchBtn.click(); return;
        }
        
        if (isModalVisible) {
            const isObjectFilterModalVisible = objectFilterModal.style.display === 'flex';
            if (isObjectFilterModalVisible && !isTyping) {
                if (event.key in LABEL_SHORTCUTS && drawnBoxes.length > 0) { event.preventDefault(); drawnBoxes[drawnBoxes.length - 1].label = LABEL_SHORTCUTS[event.key]; }
                else if (event.key === '7' && drawnBoxes.length > 0) { event.preventDefault(); const customLabel = prompt("Enter custom object label:", "person"); if (customLabel) drawnBoxes[drawnBoxes.length - 1].label = customLabel.toLowerCase(); }
                else if (event.key === 'Backspace' && drawnBoxes.length > 0) { event.preventDefault(); drawnBoxes.pop(); }
                else if (event.key.toLowerCase() === 'c') { event.preventDefault(); if (confirm("Clear all drawn boxes?")) drawnBoxes = []; }
                redrawCanvas(); updateDrawnBoxesList();
            }
            return;
        }

        const stageToModify = isTyping ? activeElement.closest('.stage-card') : stagesContainer.querySelector('.stage-card:last-child');

        if ((event.ctrlKey || event.metaKey) && event.altKey && !event.shiftKey) {
            let handled = true;
            if (stageToModify) {
                switch(event.key.toLowerCase()) {
                    case 'u': stageToModify.querySelector('.type-btn[data-type="text"]')?.click(); break;
                    case 'i': stageToModify.querySelector('.type-btn[data-type="image"]')?.click(); break;
                    case 'o': stageToModify.querySelector('.type-btn[data-type="ocr"]')?.click(); break;
                    case 'p': stageToModify.querySelector('.type-btn[data-type="asr"]')?.click(); break;
                    case 'g': stageToModify.querySelector('.type-btn[data-type="gen_image"]')?.click(); break;
                    case 'm': stageToModify.querySelector('.mic-btn')?.click(); break;
                    case 'k': onlyMetaBtn.click(); break;

                    default: handled = false;
                }
            } else { 
                // START: ADD THIS BLOCK FOR WHEN NOT TYPING
                if (event.key.toLowerCase() === 'k') {
                    onlyMetaBtn.click();
                } else {
                    handled = false;
                }
                // END: ADD THIS BLOCK
            }
            if(handled) { event.preventDefault(); return; }
        }
        else if ((event.ctrlKey || event.metaKey) && event.shiftKey) {
            let handled = true;
            switch (event.code) {
                case 'KeyR': document.getElementById('resetBtn')?.click(); break;
                case 'KeyG': ambiguousBtn.click(); break;
                case 'BracketRight': removeStageFromEnd(); break;
                case 'BracketLeft': removeStageFromStart(); break;
                default: handled = false;
            }
            if (handled) event.preventDefault();
        }
        else if ((event.ctrlKey || event.metaKey) && !event.altKey) {
            const targetStageNum = parseInt(event.key);
            if (!isNaN(targetStageNum) && targetStageNum >= 1 && targetStageNum <= 9) {
                event.preventDefault();
                const allStages = stagesContainer.querySelectorAll('.stage-card');
                if (targetStageNum <= allStages.length) focusOnStageInput(allStages[targetStageNum - 1]);
                return;
            }
            let handled = true;
            switch (event.code) {
                case 'BracketRight': addStageToEnd(); break;
                case 'BracketLeft': addStageToStart(); break;
                default:
                    switch (event.key.toLowerCase()) {
                        case 'g': clusterBtn.click(); break;
                        case 'q': stageToModify?.querySelector('.option-btn[data-option="enhance"]')?.click(); break;
                        case 'u': stageToModify?.querySelector('.option-btn[data-option="bge_caption"]')?.click(); break;
                        default: handled = false;
                    }
            }
            if (handled) event.preventDefault();
        }
        else if (isTyping && ['ArrowUp', 'ArrowDown'].includes(event.key)) {
            const allStages = Array.from(stagesContainer.querySelectorAll('.stage-card'));
            if (allStages.length > 1) {
                const currentStage = activeElement.closest('.stage-card');
                let currentIndex = currentStage ? allStages.indexOf(currentStage) : -1;
                if (currentIndex !== -1) {
                    event.preventDefault();
                    if (event.key === 'ArrowUp') { currentIndex = (currentIndex - 1 + allStages.length) % allStages.length; }
                    else if (event.key === 'ArrowDown') { currentIndex = (currentIndex + 1) % allStages.length; }
                    focusOnStageInput(allStages[currentIndex]);
                }
            }
        }

        if (videoPreviewModal.style.display !== 'flex') { return; }

        if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
            event.preventDefault();
            const direction = event.key === 'ArrowRight' ? 1 : -1;
            handleScrub(direction);
        }
    });
});