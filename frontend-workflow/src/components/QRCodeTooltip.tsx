import React, { useState } from 'react';

interface QRCodeTooltipProps {
  children: React.ReactNode;
}

const QRCodeTooltip: React.FC<QRCodeTooltipProps> = ({ children }) => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div 
      className="relative inline-flex items-center"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      
      {isVisible && (
        <div className="absolute top-full left-1/2 -translate-x-1/2 mt-3 w-40 p-2 bg-white rounded-lg shadow-xl z-50 border border-gray-200 animate-in fade-in zoom-in duration-200">
          <div className="flex flex-col items-center gap-1.5">
            <div className="w-32 h-32 bg-gray-100 rounded-md overflow-hidden">
               <img src="/wechat.png" alt="微信群二维码" className="w-full h-full object-contain" />
            </div>
            <p className="text-[10px] text-gray-700 text-center leading-tight font-medium">
              分享项目/加群<br/>凭截图，领取免费Key 🎁
            </p>
          </div>
          <div className="absolute bottom-full left-1/2 -translate-x-1/2 -mb-1 w-0 h-0 border-x-4 border-x-transparent border-b-[6px] border-b-white"></div>
        </div>
      )}
    </div>
  );
};

export default QRCodeTooltip;
