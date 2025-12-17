import React, { useEffect, useState } from 'react';
import ChatContainer from '../components/Chat/ChatContainer';

// This component wraps the entire app and ensures the chat is available on all pages
const Root = ({ children }: { children: React.ReactNode }) => {
  const [showChat, setShowChat] = useState(false);

  // Initialize the chat functionality once the component mounts
  useEffect(() => {
    setShowChat(true);
  }, []);

  return (
    <>
        {children}
        {showChat && <ChatContainer />}
    </>
  );
};

export default Root;