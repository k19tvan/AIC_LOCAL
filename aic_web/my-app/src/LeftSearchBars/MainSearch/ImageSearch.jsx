import React from 'react'
import { PhotoIcon } from '@heroicons/react/24/solid';

const ImageSearch = () => {
  return (
    <button className="flex-1 mx-1 aspect-square flex items-center justify-center hover:bg-amber-300">
      <PhotoIcon className="text-white w-2/3 h-2/3" />
    </button>
  );
}

export default ImageSearch
