import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import Login from './components/Login';
import Signup from './components/Signup';
import Main from './components/Main';
import Menu1 from './components/Menu1';
import Chatbot from './components/Chatbot';
import LoginMain from './components/LoginMain'
import Mypage from './components/Mypage';

const App = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const f=(i)=> {
    setIsLoggedIn(i)
  }
  return (
    <div>
      <Header isLoggedIn={isLoggedIn} />
      
      <Routes>
        <Route path="/" element={<Main />} exact /> 
        <Route path="/signup" element={<Signup />} />
        <Route path="/login" element={<Login />} />
        <Route path="/logout" element={<Main />} />
        <Route path="/LoginMain" element={<LoginMain f={f} />} />
        {/* <Route path='/logout' element={<Logout/>}/> */}
        <Route path='/mypage' element={<Mypage/>}/>
        <Route component={Menu1} path="/menu1" />
      </Routes>
      <Chatbot/>
      <Footer />
    </div>
  );
};

export default App;