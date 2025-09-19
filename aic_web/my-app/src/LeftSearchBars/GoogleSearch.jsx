import React from 'react'
import { MagnifyingGlassIcon } from '@heroicons/react/24/solid'

const GoogleSearch = () => {
  return (
    <div className='flex-[0.05] m-4 border border-white'>
      <div className='flex m-1'>
        <input 
          type="text" 
          className='bg-white text-black w-full flex-[0.8] px-4 border-[#FFD700] border font-mono'
          placeholder='Google Search'
        />

        <button 
          className='flex-[0.2] bg-[#ee9e34] flex justify-center items-center'
        >
          <MagnifyingGlassIcon className="text-white h-3/4 w-3/4" />

        </button>
      </div>
    </div>
  )
}

export default GoogleSearch
