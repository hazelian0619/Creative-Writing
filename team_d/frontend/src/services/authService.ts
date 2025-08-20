/**
 * 认证服务
 * 处理用户登录、注册、token管理等认证相关功能
 */

import { get, post } from './apiService';
import { API_ENDPOINTS } from '@/constants';
import { User } from '@/types';

// ============ 类型定义 ============

interface LoginRequest {
  username: string;
  password: string;
  rememberMe?: boolean;
}

interface LoginResponse {
  accessToken: string;
  refreshToken: string;
  user: User;
  expiresIn: number;
}

interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  profile: {
    displayName: string;
    grade?: string;
    school?: string;
    major?: string;
  };
}

interface RegisterResponse {
  accessToken: string;
  refreshToken: string;
  user: User;
  expiresIn: number;
}

interface RefreshTokenResponse {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

interface ValidateTokenResponse {
  valid: boolean;
  user: User;
  expiresIn: number;
}

// ============ 认证服务方法 ============

/**
 * 用户登录
 */
export async function login(credentials: LoginRequest): Promise<LoginResponse> {
  try {
    const response = await post<LoginResponse>(API_ENDPOINTS.AUTH.LOGIN, {
      username: credentials.username,
      password: credentials.password,
      rememberMe: credentials.rememberMe || false,
      loginTime: Date.now(),
      userAgent: navigator.userAgent,
      clientInfo: {
        platform: navigator.platform,
        language: navigator.language,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      },
    });

    if (!response.success) {
      throw new Error(response.error?.message || '登录失败');
    }

    // 记录登录成功日志
    console.log('Login successful:', {
      userId: response.data?.user.id,
      username: response.data?.user.username,
      loginTime: new Date().toISOString(),
    });

    return response.data!;
  } catch (error: any) {
    console.error('Login failed:', error);
    
    // 统一错误处理
    if (error.status === 401) {
      throw new Error('用户名或密码错误');
    } else if (error.status === 423) {
      throw new Error('账户已被锁定，请联系管理员');
    } else if (error.status === 429) {
      throw new Error('登录尝试次数过多，请稍后重试');
    } else if (error.code === 'NETWORK_ERROR') {
      throw new Error('网络连接失败，请检查网络设置');
    } else {
      throw new Error(error.message || '登录失败，请稍后重试');
    }
  }
}

/**
 * 用户注册
 */
export async function register(userData: RegisterRequest): Promise<RegisterResponse> {
  try {
    // 数据验证
    validateRegistrationData(userData);

    const response = await post<RegisterResponse>(API_ENDPOINTS.AUTH.REGISTER, {
      ...userData,
      registerTime: Date.now(),
      userAgent: navigator.userAgent,
      clientInfo: {
        platform: navigator.platform,
        language: navigator.language,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      },
    });

    if (!response.success) {
      throw new Error(response.error?.message || '注册失败');
    }

    // 记录注册成功日志
    console.log('Registration successful:', {
      userId: response.data?.user.id,
      username: response.data?.user.username,
      email: response.data?.user.email,
      registerTime: new Date().toISOString(),
    });

    return response.data!;
  } catch (error: any) {
    console.error('Registration failed:', error);
    
    // 统一错误处理
    if (error.status === 409) {
      throw new Error('用户名或邮箱已存在');
    } else if (error.status === 400) {
      throw new Error('注册信息格式错误，请检查后重试');
    } else if (error.code === 'NETWORK_ERROR') {
      throw new Error('网络连接失败，请检查网络设置');
    } else {
      throw new Error(error.message || '注册失败，请稍后重试');
    }
  }
}

/**
 * 刷新访问令牌
 */
export async function refreshToken(refreshToken: string): Promise<RefreshTokenResponse> {
  try {
    const response = await post<RefreshTokenResponse>(API_ENDPOINTS.AUTH.REFRESH, {
      refreshToken,
      timestamp: Date.now(),
    });

    if (!response.success) {
      throw new Error(response.error?.message || 'Token刷新失败');
    }

    return response.data!;
  } catch (error: any) {
    console.error('Token refresh failed:', error);
    
    if (error.status === 401) {
      throw new Error('刷新令牌已过期，请重新登录');
    } else if (error.status === 400) {
      throw new Error('刷新令牌格式错误');
    } else {
      throw new Error(error.message || 'Token刷新失败');
    }
  }
}

/**
 * 验证访问令牌
 */
export async function validateToken(token: string): Promise<ValidateTokenResponse> {
  try {
    const response = await get<ValidateTokenResponse>('/auth/validate', {
      token,
      timestamp: Date.now(),
    });

    if (!response.success) {
      throw new Error(response.error?.message || 'Token验证失败');
    }

    return response.data!;
  } catch (error: any) {
    console.error('Token validation failed:', error);
    
    if (error.status === 401) {
      throw new Error('访问令牌无效或已过期');
    } else {
      throw new Error(error.message || 'Token验证失败');
    }
  }
}

/**
 * 用户登出
 */
export async function logout(): Promise<void> {
  try {
    const response = await post(API_ENDPOINTS.AUTH.LOGOUT, {
      logoutTime: Date.now(),
    });

    if (!response.success) {
      console.warn('Logout API failed, but continuing with local cleanup');
    }

    console.log('Logout successful');
  } catch (error: any) {
    console.warn('Logout API failed:', error);
    // 即使API调用失败，也要继续本地清理
  }
}

/**
 * 修改密码
 */
export async function changePassword(data: {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
}): Promise<void> {
  try {
    // 客户端验证
    if (data.newPassword !== data.confirmPassword) {
      throw new Error('新密码和确认密码不一致');
    }

    if (data.newPassword.length < 8) {
      throw new Error('新密码长度至少为8位');
    }

    if (data.currentPassword === data.newPassword) {
      throw new Error('新密码不能与当前密码相同');
    }

    const response = await post('/auth/change-password', {
      currentPassword: data.currentPassword,
      newPassword: data.newPassword,
      timestamp: Date.now(),
    });

    if (!response.success) {
      throw new Error(response.error?.message || '修改密码失败');
    }

    console.log('Password changed successfully');
  } catch (error: any) {
    console.error('Change password failed:', error);
    
    if (error.status === 401) {
      throw new Error('当前密码错误');
    } else if (error.status === 400) {
      throw new Error('密码格式不符合要求');
    } else {
      throw new Error(error.message || '修改密码失败');
    }
  }
}

/**
 * 重置密码（忘记密码）
 */
export async function resetPassword(email: string): Promise<void> {
  try {
    const response = await post('/auth/reset-password', {
      email,
      timestamp: Date.now(),
      clientInfo: {
        userAgent: navigator.userAgent,
        language: navigator.language,
      },
    });

    if (!response.success) {
      throw new Error(response.error?.message || '密码重置失败');
    }

    console.log('Password reset email sent');
  } catch (error: any) {
    console.error('Password reset failed:', error);
    
    if (error.status === 404) {
      throw new Error('该邮箱地址未注册');
    } else if (error.status === 429) {
      throw new Error('重置请求过于频繁，请稍后重试');
    } else {
      throw new Error(error.message || '密码重置失败');
    }
  }
}

/**
 * 确认密码重置
 */
export async function confirmPasswordReset(data: {
  token: string;
  newPassword: string;
  confirmPassword: string;
}): Promise<void> {
  try {
    // 客户端验证
    if (data.newPassword !== data.confirmPassword) {
      throw new Error('新密码和确认密码不一致');
    }

    if (data.newPassword.length < 8) {
      throw new Error('新密码长度至少为8位');
    }

    const response = await post('/auth/confirm-reset', {
      token: data.token,
      newPassword: data.newPassword,
      timestamp: Date.now(),
    });

    if (!response.success) {
      throw new Error(response.error?.message || '密码重置确认失败');
    }

    console.log('Password reset confirmed');
  } catch (error: any) {
    console.error('Password reset confirmation failed:', error);
    
    if (error.status === 400) {
      throw new Error('重置令牌无效或已过期');
    } else {
      throw new Error(error.message || '密码重置确认失败');
    }
  }
}

/**
 * 检查用户名是否可用
 */
export async function checkUsernameAvailability(username: string): Promise<boolean> {
  try {
    const response = await get<{ available: boolean }>('/auth/check-username', {
      username,
    });

    if (!response.success) {
      throw new Error('检查用户名失败');
    }

    return response.data!.available;
  } catch (error: any) {
    console.error('Username check failed:', error);
    throw new Error('检查用户名失败');
  }
}

/**
 * 检查邮箱是否可用
 */
export async function checkEmailAvailability(email: string): Promise<boolean> {
  try {
    const response = await get<{ available: boolean }>('/auth/check-email', {
      email,
    });

    if (!response.success) {
      throw new Error('检查邮箱失败');
    }

    return response.data!.available;
  } catch (error: any) {
    console.error('Email check failed:', error);
    throw new Error('检查邮箱失败');
  }
}

// ============ 工具函数 ============

/**
 * 验证注册数据
 */
function validateRegistrationData(data: RegisterRequest): void {
  // 用户名验证
  if (!data.username || data.username.length < 3) {
    throw new Error('用户名长度至少为3位');
  }

  if (!/^[a-zA-Z0-9_]+$/.test(data.username)) {
    throw new Error('用户名只能包含字母、数字和下划线');
  }

  // 邮箱验证
  if (!data.email) {
    throw new Error('邮箱地址不能为空');
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(data.email)) {
    throw new Error('邮箱地址格式不正确');
  }

  // 密码验证
  if (!data.password || data.password.length < 8) {
    throw new Error('密码长度至少为8位');
  }

  if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(data.password)) {
    throw new Error('密码必须包含大小写字母和数字');
  }

  // 个人资料验证
  if (!data.profile.displayName || data.profile.displayName.trim().length === 0) {
    throw new Error('显示名称不能为空');
  }

  if (data.profile.displayName.length > 50) {
    throw new Error('显示名称长度不能超过50字符');
  }
}

/**
 * 检查密码强度
 */
export function checkPasswordStrength(password: string): {
  score: number; // 0-100
  level: 'weak' | 'medium' | 'strong' | 'very-strong';
  suggestions: string[];
} {
  let score = 0;
  const suggestions: string[] = [];

  // 长度检查
  if (password.length >= 8) {
    score += 20;
  } else {
    suggestions.push('密码长度至少为8位');
  }

  if (password.length >= 12) {
    score += 10;
  }

  // 字符类型检查
  if (/[a-z]/.test(password)) {
    score += 15;
  } else {
    suggestions.push('包含小写字母');
  }

  if (/[A-Z]/.test(password)) {
    score += 15;
  } else {
    suggestions.push('包含大写字母');
  }

  if (/\d/.test(password)) {
    score += 15;
  } else {
    suggestions.push('包含数字');
  }

  if (/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) {
    score += 15;
  } else {
    suggestions.push('包含特殊字符');
  }

  // 复杂度检查
  const uniqueChars = new Set(password).size;
  if (uniqueChars >= password.length * 0.7) {
    score += 10;
  }

  // 确定等级
  let level: 'weak' | 'medium' | 'strong' | 'very-strong';
  if (score >= 90) {
    level = 'very-strong';
  } else if (score >= 70) {
    level = 'strong';
  } else if (score >= 50) {
    level = 'medium';
  } else {
    level = 'weak';
  }

  return { score, level, suggestions };
}

/**
 * 获取当前用户的访问令牌
 */
export function getCurrentToken(): string | null {
  return localStorage.getItem('creative_writing_access_token');
}

/**
 * 检查用户是否已登录
 */
export function isAuthenticated(): boolean {
  const token = getCurrentToken();
  if (!token) return false;

  try {
    // 简单的JWT过期检查（不验证签名）
    const payload = JSON.parse(atob(token.split('.')[1]));
    const expiry = payload.exp * 1000; // 转换为毫秒
    return Date.now() < expiry;
  } catch {
    return false;
  }
}

/**
 * 获取令牌剩余有效时间（秒）
 */
export function getTokenRemainingTime(): number {
  const token = getCurrentToken();
  if (!token) return 0;

  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    const expiry = payload.exp * 1000;
    const remaining = Math.max(0, expiry - Date.now());
    return Math.floor(remaining / 1000);
  } catch {
    return 0;
  }
}