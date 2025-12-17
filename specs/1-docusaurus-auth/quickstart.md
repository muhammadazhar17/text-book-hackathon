# Quickstart: Docusaurus Frontend with Better Auth Authentication

## Prerequisites
- Node.js 18+ installed
- npm or yarn package manager
- Basic knowledge of React and TypeScript
- Docusaurus project already set up

## Installation Steps

### 1. Install Better Auth Dependencies
```bash
cd physical-ai-robotics-docs
npm install @better-auth/react @better-auth/node
```

### 2. Configure Better Auth Client
Create `src/services/auth-client.ts`:
```typescript
import { createAuthClient } from "@better-auth/react";

export const authClient = createAuthClient({
  baseURL: process.env.REACT_APP_BETTER_AUTH_URL || "http://localhost:8000",
  // Add other configuration as needed
});
```

### 3. Set Up Environment Variables
Add to `.env` file in the `physical-ai-robotics-docs` directory:
```
REACT_APP_BETTER_AUTH_URL=http://localhost:8000
```

### 4. Create Authentication Context
Create `src/context/AuthContext.tsx`:
```typescript
import React, { createContext, useContext, useEffect, useState } from 'react';
import { authClient } from '../services/auth-client';
import { useSession } from '@better-auth/react';

interface AuthContextType {
  user: any;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, name: string, hardwareBackground?: string, softwareBackground?: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { session, signIn, signOut, signUp } = useSession();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Initialize auth state
    setIsLoading(false);
  }, []);

  const login = async (email: string, password: string) => {
    // Implementation using Better Auth signIn
  };

  const signup = async (email: string, password: string, name: string, hardwareBackground?: string, softwareBackground?: string) => {
    // Implementation using Better Auth signUp
  };

  const logout = async () => {
    await signOut();
  };

  return (
    <AuthContext.Provider value={{
      user: session?.user,
      isLoading,
      isAuthenticated: !!session,
      login,
      signup,
      logout
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
```

### 5. Update Docusaurus Configuration
Modify `docusaurus.config.ts` to include authentication components in the navbar.

### 6. Create Authentication Components
- Create `LoginForm.tsx`
- Create `SignupForm.tsx`
- Create `ProfileDropdown.tsx`

## Running the Application
```bash
cd physical-ai-robotics-docs
npm start
```

## Testing Authentication
1. Visit the site and verify "Sign Up" and "Sign In" buttons appear in the navbar
2. Test the signup flow with valid credentials
3. Test the login flow with registered credentials
4. Verify navbar updates to show profile and sign out
5. Test the sign out functionality
6. Verify the RAG chatbot works in both authenticated and unauthenticated states