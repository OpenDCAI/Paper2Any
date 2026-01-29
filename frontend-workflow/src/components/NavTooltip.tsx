import React, { useState, useRef, useEffect } from 'react';

interface NavTooltipProps {
  children: React.ReactNode;
  content: string;
}

const NavTooltip: React.FC<NavTooltipProps> = ({ children, content }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const buttonRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isVisible && buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setPosition({
        top: rect.top + rect.height / 2,
        left: rect.right + 12
      });
    }
  }, [isVisible]);

  return (
    <>
      <div
        ref={buttonRef}
        className="relative inline-flex w-full"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
      >
        {children}
      </div>

      {isVisible && (
        <div
          className="fixed w-64 p-3 bg-white rounded-lg shadow-xl z-[9999] border border-gray-200 animate-in fade-in zoom-in duration-200"
          style={{
            top: `${position.top}px`,
            left: `${position.left}px`,
            transform: 'translateY(-50%)'
          }}
        >
          <p className="text-xs text-gray-700 leading-relaxed">
            {content}
          </p>
          {/* 左侧三角箭头 */}
          <div className="absolute right-full mr-[-6px] top-1/2 -translate-y-1/2 w-0 h-0 border-y-4 border-y-transparent border-r-[6px] border-r-white"></div>
        </div>
      )}
    </>
  );
};

export default NavTooltip;
