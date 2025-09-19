import { useState } from 'react'
import './App.css'
import Header from './TopToolBars/Header'
import LeftSearchBars from './LeftSearchBars/LeftSearchBars'
import ResultBars from './ResultBars/ResultBars'

function App() {
  return (
    <div className='flex flex-col h-screen w-screen'>
      <Header/>
      <div className='flex-1 flex bg-[#014D4D]'>
        <div className="flex-[0.2] h-full">
          <LeftSearchBars/>
        </div>
        <div className="flex-1 h-full">
          <ResultBars/>
        </div>
      </div>
    </div>
  )
}

export default App
