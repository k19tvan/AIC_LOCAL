import React from 'react'
import ResetStageButton from './ResetStageButton'
import UserInfo from './UserInfo'
import LoginButton from './LoginButton'
import ClusterButton from './ClusterButton'
import ObjectFilterButton from './ObjectFilterButton'
import ModelSelectionButton from './ModelSelectionButton'


const Header = () => {
  return (
    <div className='bg-[#024A5A] w-full h-16 px-4 flex items-center justify-between bg-gradient-to-r from-[#1D293D] to-[#024A5A] border-b-black border'>
        <div>
            {/* <UserInfo/> */}
        </div>

        <div className="flex items-center gap-4 justify-end">
            <ModelSelectionButton/>
            <ObjectFilterButton/>
            <ClusterButton/>
            <ResetStageButton/>
            <LoginButton/>
        </div>
    </div>
  )
}

export default Header
