import React from 'react'
import { ChatBubbleBottomCenterIcon } from '@heroicons/react/24/solid';

const TextSearch = () => {
  return (
    <button className="flex-1 mx-1 aspect-square flex items-center justify-center hover:bg-amber-300">
      <ChatBubbleBottomCenterIcon className="text-white w-2/3 h-2/3" />
    </button>
  );
}

export default TextSearch
