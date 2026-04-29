CREATE DATABASE GAME
GO
CREATE TABLE NhanVat(
    maNV varchar(5) PRIMARY KEY,
    TenNV nvarchar(50) NOT NULL UNIQUE, -- Không được để trống và trùng
    HePhai nvarchar(20) DEFAULT 'Novice',
    CapDo int CHECK (CapDo > 0)
)

CREATE TABLE VuKhi(
    MaVK varchar(8)PRIMARY KEY,
    TenVK nvarchar(50) NOT NULL UNIQUE,
    MaNv_SoHuu varchar(8),
    CONSTRAINT FK_VK_NH FOREIGN KEY (MaNv_SoHuu REFERENCES NhanVat(maNV)
)

ALTER TABLE VuKhi)
ADD Dame int,
ADD CONSTRAINT CK_SucManh CHECK(Dame > 0);


CREATE TABLE TrangThai(
    MaNV varchar(5),
    CONSTRAINT FK_TrangThai_NV FOREIGN KEY (MaNV) REFERENCES NhanVat(maNV),
    HPToiDa int,
    HPHienTai int,
    CONSTRAINT CK_KiemTraHP CHECK (HPHienTai <= HPToiDa AND HPHienTaij >= 0)
)