// End-to-end tests for the chat functionality
// This is a simplified test file to demonstrate the concept

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChatContainer from '../ChatContainer';

// Mock the API client to avoid actual network calls during testing
jest.mock('../../services/api-client', () => ({
  __esModule: true,
  default: {
    sendChatMessage: jest.fn().mockResolvedValue({
      response: 'This is a test response from the AI assistant.',
      sources: ['test-source.md']
    }),
    checkBackendStatus: jest.fn().mockResolvedValue(true)
  }
}));

describe('ChatContainer E2E Tests', () => {
  test('should open chat panel when button is clicked', () => {
    render(<ChatContainer />);

    // Find and click the chat button
    const chatButton = screen.getByLabelText('Open chat');
    fireEvent.click(chatButton);

    // Verify that the chat panel opens
    expect(screen.getByLabelText('Close chat')).toBeInTheDocument();
    expect(screen.getByText('AI Assistant')).toBeInTheDocument();
  });

  test('should send a message and receive a response', async () => {
    render(<ChatContainer />);

    // Open the chat panel
    const chatButton = screen.getByLabelText('Open chat');
    fireEvent.click(chatButton);

    // Find the message input and send a message
    const messageInput = screen.getByPlaceholderText('Type your message here...');
    fireEvent.change(messageInput, { target: { value: 'Hello, test message!' } });

    // Submit the message
    const sendButton = screen.getByLabelText('Send message');
    fireEvent.click(sendButton);

    // Wait for the response to appear
    await waitFor(() => {
      expect(screen.getByText('Hello, test message!')).toBeInTheDocument();
      expect(screen.getByText('This is a test response from the AI assistant.')).toBeInTheDocument();
    });
  });

  test('should display sources when provided in response', async () => {
    render(<ChatContainer />);

    // Open the chat panel
    const chatButton = screen.getByLabelText('Open chat');
    fireEvent.click(chatButton);

    // Find the message input and send a message
    const messageInput = screen.getByPlaceholderText('Type your message here...');
    fireEvent.change(messageInput, { target: { value: 'Test message with sources' } });

    // Submit the message
    const sendButton = screen.getByLabelText('Send message');
    fireEvent.click(sendButton);

    // Wait for the response with sources to appear
    await waitFor(() => {
      expect(screen.getByText('Sources:')).toBeInTheDocument();
      expect(screen.getByText('test-source.md')).toBeInTheDocument();
    });
  });

  test('should handle error responses gracefully', async () => {
    // Mock an error response
    const mockApiClient = require('../../services/api-client').default;
    mockApiClient.sendChatMessage.mockResolvedValueOnce({ error: 'Test error message' });

    render(<ChatContainer />);

    // Open the chat panel
    const chatButton = screen.getByLabelText('Open chat');
    fireEvent.click(chatButton);

    // Find the message input and send a message
    const messageInput = screen.getByPlaceholderText('Type your message here...');
    fireEvent.change(messageInput, { target: { value: 'Message that will error' } });

    // Submit the message
    const sendButton = screen.getByLabelText('Send message');
    fireEvent.click(sendButton);

    // Wait for the error response to appear
    await waitFor(() => {
      expect(screen.getByText('Test error message')).toBeInTheDocument();
    });
  });
});