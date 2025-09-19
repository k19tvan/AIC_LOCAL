import React from 'react'
import { MicrophoneIcon } from '@heroicons/react/24/solid';

const AsrSearch = ({ enabled, onToggle }) => {
  return (
    <button
      onClick={onToggle}
      className={`flex-1 mx-1 aspect-square flex items-center justify-center 
        ${enabled ? 'bg-amber-400' : 'hover:bg-amber-300'}`}
    >
      <MicrophoneIcon className="text-white w-2/3 h-2/3" />
    </button>
  );
}

export default AsrSearch
