import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import styled from 'styled-components';
import { Menu, X } from 'lucide-react';

const HeaderContainer = styled.header`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid ${props => props.theme.colors.border};
  z-index: 1000;
  transition: all 0.2s ease;
`;

const HeaderContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 ${props => props.theme.spacing[4]};
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 64px;
`;

const Logo = styled(Link)`
  font-size: ${props => props.theme.fontSizes.xl};
  font-weight: ${props => props.theme.fontWeights.bold};
  color: ${props => props.theme.colors.primary};
  text-decoration: none;
  margin-right: auto;
  
  &:hover {
    color: ${props => props.theme.colors.primary};
  }
`;

const Nav = styled.nav`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing[8]};
  margin-left: auto;

  @media (max-width: ${props => props.theme.breakpoints.md}) {
    display: none;
  }
`;

const NavLink = styled(Link)`
  color: ${props => props.isActive ? props.theme.colors.primary : props.theme.colors.text.secondary};
  text-decoration: none;
  font-weight: ${props => props.isActive ? props.theme.fontWeights.medium : props.theme.fontWeights.normal};
  transition: color 0.2s ease;
  position: relative;

  &:hover {
    color: ${props => props.theme.colors.primary};
  }

  &::after {
    content: '';
    position: absolute;
    bottom: -4px;
    left: 0;
    right: 0;
    height: 2px;
    background: ${props => props.isActive ? props.theme.colors.primary : 'transparent'};
    transition: background 0.2s ease;
  }
`;

const MobileMenuButton = styled.button`
  display: none;
  background: none;
  border: none;
  color: ${props => props.theme.colors.text.primary};
  cursor: pointer;
  padding: ${props => props.theme.spacing[2]};

  @media (max-width: ${props => props.theme.breakpoints.md}) {
    display: block;
  }
`;

const MobileMenu = styled.div`
  position: fixed;
  top: 64px;
  left: 0;
  right: 0;
  background: ${props => props.theme.colors.background};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing[4]};
  transform: translateY(${props => props.isOpen ? '0' : '-100%'});
  transition: transform 0.3s ease;
  z-index: 999;

  @media (min-width: ${props => props.theme.breakpoints.md}) {
    display: none;
  }
`;

const MobileNavLink = styled(Link)`
  display: block;
  padding: ${props => props.theme.spacing[3]} 0;
  color: ${props => props.isActive ? props.theme.colors.accent : props.theme.colors.text.secondary};
  text-decoration: none;
  font-weight: ${props => props.isActive ? props.theme.fontWeights.medium : props.theme.fontWeights.normal};
  border-bottom: 1px solid ${props => props.theme.colors.border};

  &:hover {
    color: ${props => props.theme.colors.accent};
  }
`;

const Header = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false);
  };

  const navItems = [
    { path: '/', label: 'Home' },
    { path: '/docs', label: 'Documentation' },
    { path: '/demo', label: 'Demo' }
  ];

  return (
    <>
      <HeaderContainer>
        <HeaderContent>
          <Logo to="/">Arabic Sentiment</Logo>
          
          <Nav>
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                isActive={location.pathname === item.path}
              >
                {item.label}
              </NavLink>
            ))}
          </Nav>

          <MobileMenuButton onClick={toggleMobileMenu}>
            {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </MobileMenuButton>
        </HeaderContent>
      </HeaderContainer>

      <MobileMenu isOpen={isMobileMenuOpen}>
        {navItems.map((item) => (
          <MobileNavLink
            key={item.path}
            to={item.path}
            isActive={location.pathname === item.path}
            onClick={closeMobileMenu}
          >
            {item.label}
          </MobileNavLink>
        ))}
      </MobileMenu>
    </>
  );
};

export default Header;

