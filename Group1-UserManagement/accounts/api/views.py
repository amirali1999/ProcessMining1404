from django.contrib.auth import authenticate, login, logout, get_user_model
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.authtoken.models import Token

from .serializers import RegisterSerializer, UserSerializer, LoginSerializer, RoleSerializer
from .permissions import IsAdminRole
from accounts.models import Role

User = get_user_model()


class RegisterAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            login(request, user)  # session based login
            token, _ = Token.objects.get_or_create(user=user)
            return Response({"user": UserSerializer(user).data, "token": token.key}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        username = serializer.validated_data["username"]
        password = serializer.validated_data["password"]
        user = authenticate(request, username=username, password=password)
        if not user:
            return Response({"detail": "نام کاربری یا رمز عبور نادرست است."}, status=status.HTTP_401_UNAUTHORIZED)
        login(request, user)
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"user": UserSerializer(user).data, "token": token.key})


class LogoutAPIView(APIView):
    def post(self, request):
        # Optional: delete user's token (force new login for API usage)
        Token.objects.filter(user=request.user).delete()
        logout(request)
        return Response({"detail": "خروج انجام شد."})


class MeAPIView(APIView):
    def get(self, request):
        return Response(UserSerializer(request.user).data)


class UsersListAPIView(APIView):
    permission_classes = [IsAuthenticated, IsAdminRole]

    def get(self, request):
        qs = User.objects.all().order_by("id")
        return Response(UserSerializer(qs, many=True).data)


class RolesListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        roles = Role.objects.all().order_by("id")
        return Response(RoleSerializer(roles, many=True).data)
