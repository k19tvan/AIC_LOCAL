import React from 'react'
import GoogleSearch from './GoogleSearch'
import MainSearch from './MainSearch/MainSearch'

const LeftSearchBars = () => {
  return (
    <div className='h-full w-full from-[#1D293D] to-[#1D293D] flex flex-col bg-gradient-to-b'>
      <GoogleSearch/>
      <MainSearch/>
    </div>

  )
}

export default LeftSearchBars
