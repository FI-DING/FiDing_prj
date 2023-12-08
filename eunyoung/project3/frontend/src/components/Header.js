import React from 'react';
import './util.css';
import './login.css';
import Navigation from './Navigration';
// import Navigation from './Navigation';

function Header() {
  return (
    <div>
        <header>
          <nav>
            <Navigation/>
          </nav>
        </header>
    </div>
  );
}

export default Header;