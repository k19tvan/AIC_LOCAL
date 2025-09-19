import React from 'react';

const ResetStageButton = () => {
  return (
    <button
      className="
        bg-[#13293D]                    
        text-amber-400                          
        border border-amber-500/50              
        font-bold
        py-1 px-3                               
        tracking-wider                          
        shadow-lg
        font-mono
        hover:bg-slate-700
        hover:shadow-xl
        transition-all duration-300 ease-in-out 
      "
    >
      Reset Stage
    </button>
  );
};

export default ResetStageButton;