import React from 'react';
import { Bars3Icon } from '@heroicons/react/24/solid';

const OcrSearch = ({ enabled, onToggle }) => {
  return (
    <button
      onClick={onToggle}
      className={`flex-1 mx-1 aspect-square flex items-center justify-center 
        ${enabled ? 'bg-amber-400' : 'hover:bg-amber-300'}`}
    >
      <Bars3Icon className="text-white w-2/3 h-2/3" />
    </button>
  );
};

export default OcrSearch;