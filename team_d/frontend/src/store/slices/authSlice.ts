/**
 * 认证状态管理 Slice
 * 处理用户登录、登出、token管理等认证相关状态
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { User } from '@/types';
import { STORAGE_KEYS } from '@/constants';
import * as authService from '@/services/authService';

// 状态类型定义
interface AuthState {
  isAuthenticated: boolean;
  token: string | null;
  refreshToken: string | null;
  user: User | null;
  loading: boolean;
  error: string | null;
  lastLoginAt: string | null;
}

// 初始状态
const initialState: AuthState = {
  isAuthenticated: false,
  token: localStorage.getItem(STORAGE_KEYS.ACCESS_TOKEN),
  refreshToken: localStorage.getItem(STORAGE_KEYS.REFRESH_TOKEN),
  user: JSON.parse(localStorage.getItem(STORAGE_KEYS.USER_INFO) || 'null'),
  loading: false,
  error: null,
  lastLoginAt: null,
};

// 如果有token则设置为已认证状态
if (initialState.token && initialState.user) {
  initialState.isAuthenticated = true;
}

// ============ 异步 Actions ============

/** 用户登录 */
export const loginAsync = createAsyncThunk(
  'auth/login',
  async (credentials: { username: string; password: string; rememberMe?: boolean }) => {
    const response = await authService.login(credentials);
    
    // 保存到本地存储
    localStorage.setItem(STORAGE_KEYS.ACCESS_TOKEN, response.accessToken);
    localStorage.setItem(STORAGE_KEYS.REFRESH_TOKEN, response.refreshToken);
    localStorage.setItem(STORAGE_KEYS.USER_INFO, JSON.stringify(response.user));
    
    return response;
  }
);

/** 用户注册 */
export const registerAsync = createAsyncThunk(
  'auth/register',
  async (userData: {
    username: string;
    email: string;
    password: string;
    profile: {
      displayName: string;
      grade?: string;
      school?: string;
      major?: string;
    };
  }) => {
    const response = await authService.register(userData);
    
    // 注册成功后自动登录
    localStorage.setItem(STORAGE_KEYS.ACCESS_TOKEN, response.accessToken);
    localStorage.setItem(STORAGE_KEYS.REFRESH_TOKEN, response.refreshToken);
    localStorage.setItem(STORAGE_KEYS.USER_INFO, JSON.stringify(response.user));
    
    return response;
  }
);

/** 刷新Token */
export const refreshTokenAsync = createAsyncThunk(
  'auth/refreshToken',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: AuthState };
      const refreshToken = state.auth.refreshToken;
      
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }
      
      const response = await authService.refreshToken(refreshToken);
      
      // 更新本地存储
      localStorage.setItem(STORAGE_KEYS.ACCESS_TOKEN, response.accessToken);
      localStorage.setItem(STORAGE_KEYS.REFRESH_TOKEN, response.refreshToken);
      
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  }
);

/** 用户登出 */
export const logoutAsync = createAsyncThunk(
  'auth/logout',
  async (_, { getState }) => {
    const state = getState() as { auth: AuthState };
    
    try {
      // 调用登出API
      if (state.auth.token) {
        await authService.logout();
      }
    } catch (error) {
      // 即使API调用失败也要清除本地状态
      console.warn('Logout API failed:', error);
    }
    
    // 清除本地存储
    localStorage.removeItem(STORAGE_KEYS.ACCESS_TOKEN);
    localStorage.removeItem(STORAGE_KEYS.REFRESH_TOKEN);
    localStorage.removeItem(STORAGE_KEYS.USER_INFO);
    
    return {};
  }
);

/** 验证当前Token */
export const validateTokenAsync = createAsyncThunk(
  'auth/validateToken',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: AuthState };
      const token = state.auth.token;
      
      if (!token) {
        throw new Error('No token to validate');
      }
      
      const response = await authService.validateToken(token);
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  }
);

// ============ Slice 定义 ============

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    // 清除错误状态
    clearError: (state) => {
      state.error = null;
    },
    
    // 更新用户信息
    updateUser: (state, action: PayloadAction<Partial<User>>) => {
      if (state.user) {
        state.user = { ...state.user, ...action.payload };
        localStorage.setItem(STORAGE_KEYS.USER_INFO, JSON.stringify(state.user));
      }
    },
    
    // 设置loading状态
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    
    // 手动登出（不调用API）
    logoutLocal: (state) => {
      state.isAuthenticated = false;
      state.token = null;
      state.refreshToken = null;
      state.user = null;
      state.error = null;
      state.lastLoginAt = null;
      
      // 清除本地存储
      localStorage.removeItem(STORAGE_KEYS.ACCESS_TOKEN);
      localStorage.removeItem(STORAGE_KEYS.REFRESH_TOKEN);
      localStorage.removeItem(STORAGE_KEYS.USER_INFO);
    },
  },
  extraReducers: (builder) => {
    // ====== 登录 ======
    builder
      .addCase(loginAsync.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(loginAsync.fulfilled, (state, action) => {
        state.loading = false;
        state.isAuthenticated = true;
        state.token = action.payload.accessToken;
        state.refreshToken = action.payload.refreshToken;
        state.user = action.payload.user;
        state.error = null;
        state.lastLoginAt = new Date().toISOString();
      })
      .addCase(loginAsync.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || '登录失败';
        state.isAuthenticated = false;
        state.token = null;
        state.refreshToken = null;
        state.user = null;
      });
    
    // ====== 注册 ======
    builder
      .addCase(registerAsync.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(registerAsync.fulfilled, (state, action) => {
        state.loading = false;
        state.isAuthenticated = true;
        state.token = action.payload.accessToken;
        state.refreshToken = action.payload.refreshToken;
        state.user = action.payload.user;
        state.error = null;
        state.lastLoginAt = new Date().toISOString();
      })
      .addCase(registerAsync.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || '注册失败';
      });
    
    // ====== 刷新Token ======
    builder
      .addCase(refreshTokenAsync.pending, (state) => {
        state.loading = true;
      })
      .addCase(refreshTokenAsync.fulfilled, (state, action) => {
        state.loading = false;
        state.token = action.payload.accessToken;
        state.refreshToken = action.payload.refreshToken;
        state.error = null;
      })
      .addCase(refreshTokenAsync.rejected, (state) => {
        state.loading = false;
        state.isAuthenticated = false;
        state.token = null;
        state.refreshToken = null;
        state.user = null;
        state.error = 'Token刷新失败，请重新登录';
        
        // 清除本地存储
        localStorage.removeItem(STORAGE_KEYS.ACCESS_TOKEN);
        localStorage.removeItem(STORAGE_KEYS.REFRESH_TOKEN);
        localStorage.removeItem(STORAGE_KEYS.USER_INFO);
      });
    
    // ====== 登出 ======
    builder
      .addCase(logoutAsync.pending, (state) => {
        state.loading = true;
      })
      .addCase(logoutAsync.fulfilled, (state) => {
        state.loading = false;
        state.isAuthenticated = false;
        state.token = null;
        state.refreshToken = null;
        state.user = null;
        state.error = null;
        state.lastLoginAt = null;
      })
      .addCase(logoutAsync.rejected, (state) => {
        state.loading = false;
        // 即使登出失败也要清除本地状态
        state.isAuthenticated = false;
        state.token = null;
        state.refreshToken = null;
        state.user = null;
        state.error = null;
        state.lastLoginAt = null;
      });
    
    // ====== 验证Token ======
    builder
      .addCase(validateTokenAsync.pending, (state) => {
        state.loading = true;
      })
      .addCase(validateTokenAsync.fulfilled, (state, action) => {
        state.loading = false;
        state.isAuthenticated = true;
        state.user = action.payload.user;
        state.error = null;
      })
      .addCase(validateTokenAsync.rejected, (state) => {
        state.loading = false;
        state.isAuthenticated = false;
        state.token = null;
        state.refreshToken = null;
        state.user = null;
        state.error = 'Token验证失败';
        
        // 清除本地存储
        localStorage.removeItem(STORAGE_KEYS.ACCESS_TOKEN);
        localStorage.removeItem(STORAGE_KEYS.REFRESH_TOKEN);
        localStorage.removeItem(STORAGE_KEYS.USER_INFO);
      });
  },
});

// ============ 导出 Actions 和 Selectors ============

export const {
  clearError,
  updateUser,
  setLoading,
  logoutLocal,
} = authSlice.actions;

// Selectors
export const selectAuth = (state: { auth: AuthState }) => state.auth;
export const selectIsAuthenticated = (state: { auth: AuthState }) => state.auth.isAuthenticated;
export const selectCurrentUser = (state: { auth: AuthState }) => state.auth.user;
export const selectAuthLoading = (state: { auth: AuthState }) => state.auth.loading;
export const selectAuthError = (state: { auth: AuthState }) => state.auth.error;
export const selectToken = (state: { auth: AuthState }) => state.auth.token;

export default authSlice.reducer;