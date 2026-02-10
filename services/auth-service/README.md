# Auth Service

JWT-based authentication and authorization service for CognitionOS.

## Features

- User registration and login
- JWT token generation (access + refresh)
- Role-based access control (RBAC)
- Password hashing with bcrypt
- Session management
- Password reset flows

## API Endpoints

### Public Endpoints

- `POST /register` - Register new user
- `POST /login` - Login and get JWT tokens
- `POST /refresh` - Refresh access token using refresh token
- `POST /forgot-password` - Request password reset
- `POST /reset-password` - Reset password with token

### Authenticated Endpoints

- `GET /me` - Get current user info
- `POST /logout` - Logout (invalidate tokens)
- `PUT /me` - Update user profile
- `PUT /me/password` - Change password

## Environment Variables

```
AUTH_SERVICE_URL=http://auth-service:8001
DATABASE_URL=postgresql://user:pass@localhost:5432/cognition
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
BCRYPT_ROUNDS=12
```

## Security

- Passwords hashed with bcrypt (12 rounds)
- JWT tokens with short expiration
- Refresh tokens stored in Redis with TTL
- Rate limiting on login attempts
- User isolation enforced at database level

## Tech Stack

- FastAPI for REST API
- PostgreSQL for user data
- Redis for session management
- bcrypt for password hashing
- PyJWT for token generation
