import React, { useState, useEffect, useRef } from 'react';
import OcrSearch from './OcrSearch';
import TextSearch from './TextSearch';
import ImageSearch from './ImageSearch';
import AsrSearch from './AsrSearch';

const MainSearch = () => {
  const [value, setValue] = useState("");

  const [ocrEnabled, setOcrEnabled] = useState(false);
  const [ocrValue, setOcrValue] = useState("");

  const [asrEnabled, setAsrEnabled] = useState(false);
  const [asrValue, setAsrValue] = useState("");

  const ocrInputRef = useRef(null);
  const asrInputRef = useRef(null);
  const mainInputRef = useRef(null);

  const handleSearch = async () => {
    console.log("handleSearch called");

    const payload = {
      query: value,
      filters: {}
    };

    if (ocrEnabled) {
      payload.filters.ocr = { enabled: true, text: ocrValue };
    }
    else {
      payload.filters.ocr = { enabled: false, text: "" };
    }

    if (asrEnabled) {
      payload.filters.asr = { enabled: true, text: asrValue };
    }
    else{
      payload.filters.asr = { enabled: false, text: "" };
    }

    if (payload.filters.asr.text == "") payload.filters.asr.enabled = false;
    if (payload.filters.ocr.text == "") payload.filters.ocr.enabled = false;

    console.log("Sending payload:", payload);

    try {
      const response = await fetch("http://localhost:18028/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log("Received result:", result);
    } catch (error) {
      console.error("Error sending request:", error);
    }
  };

  useEffect(() => {
    if (ocrEnabled && ocrInputRef.current) ocrInputRef.current.focus();
  }, [ocrEnabled]);

  useEffect(() => {
    if (asrEnabled && asrInputRef.current) asrInputRef.current.focus();
  }, [asrEnabled]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey && e.altKey && (e.key === 'o' || e.key === 'O')) {
        setOcrEnabled(prev => !prev);
        e.preventDefault();
      }
      if (e.ctrlKey && e.altKey && (e.key === 'p' || e.key === 'P')) {
        setAsrEnabled(prev => !prev);
        e.preventDefault();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className='m-4 border border-white flex flex-col text-white bg-[#0d0b38] p-2'>
      {/* Thanh icon */}
      <div className="flex justify-evenly mb-2 h-12">
        <TextSearch/>
        <ImageSearch/>
        <OcrSearch enabled={ocrEnabled} onToggle={() => setOcrEnabled(prev => !prev)} />
        <AsrSearch enabled={asrEnabled} onToggle={() => setAsrEnabled(prev => !prev)} />
      </div>

      {/* Input chính */}
      <input
        ref={mainInputRef}
        type="text"
        className="border border-white px-4 py-2 font-mono mb-2"
        placeholder="Typing query here"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') handleSearch();
        }}
      />

      {/* Inputs OCR / ASR */}
      {ocrEnabled && (
        <input
          ref={ocrInputRef}
          type="text"
          className="border border-amber-400 px-4 py-2 font-mono mb-2"
          placeholder="OCR filter"
          value={ocrValue}
          onChange={(e) => setOcrValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSearch();
          }}
        />
      )}

      {asrEnabled && (
        <input
          ref={asrInputRef}
          type="text"
          className="border border-amber-400 px-4 py-2 font-mono mb-2"
          placeholder="ASR filter"
          value={asrValue}
          onChange={(e) => setAsrValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSearch();
          }}
        />
      )}

      {/* Nút hành động */}
      <div className="flex gap-2 mt-auto">
        <button
          onClick={handleSearch}
          className="flex-1 bg-amber-400 border border-white font-mono hover:bg-amber-500"
        >
          Search
        </button>
        <button className="flex-1 bg-amber-400 border border-white font-mono hover:bg-amber-500">
          Enhance
        </button>
        <button className="flex-1 bg-amber-400 border border-white font-mono hover:bg-amber-500">
          Caption
        </button>
      </div>
    </div>
  );
};

export default MainSearch;
