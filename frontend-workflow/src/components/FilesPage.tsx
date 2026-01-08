/**
 * FilesPage component showing user's generated files.
 *
 * Displays files in a table with download and delete actions.
 */

import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { getFileRecords, deleteFileRecord, FileRecord } from "../services/fileService";
import { FileText, Download, Trash2, RefreshCw, Loader2 } from "lucide-react";

function formatSize(bytes: number | null | undefined): string {
  if (!bytes) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(dateStr: string | undefined, locale: string): string {
  if (!dateStr) return "-";
  return new Date(dateStr).toLocaleDateString(locale, {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function FilesPage() {
  const { t, i18n } = useTranslation("common");
  const [files, setFiles] = useState<FileRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);

  const loadFiles = async () => {
    setLoading(true);
    try {
      const data = await getFileRecords();
      setFiles(data);
    } catch (e) {
      console.error("Failed to load files:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadFiles();
  }, []);

  const handleDelete = async (id: string, fileName: string) => {
    if (!confirm(t("filesPage.actions.confirmDelete", { fileName }))) return;

    setDeleting(id);
    try {
      const success = await deleteFileRecord(id);
      if (success) {
        setFiles(files.filter((f) => f.id !== id));
      } else {
        alert(t("filesPage.actions.deleteError"));
      }
    } catch (e) {
      console.error("Failed to delete file:", e);
      alert(t("filesPage.actions.deleteError"));
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div className="h-full overflow-auto">
      <div className="p-6 max-w-5xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-xl font-bold text-white">{t("filesPage.title")}</h1>
          <button
            onClick={loadFiles}
            disabled={loading}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw
              size={18}
              className={`text-gray-400 ${loading ? "animate-spin" : ""}`}
            />
          </button>
        </div>

        {/* Content */}
        {loading && files.length === 0 ? (
          <div className="text-center py-12">
            <Loader2
              size={32}
              className="animate-spin text-primary-500 mx-auto"
            />
            <p className="text-gray-400 mt-3">{t("filesPage.loading")}</p>
          </div>
        ) : files.length === 0 ? (
          <div className="text-center py-12 glass-dark rounded-xl border border-white/10">
            <FileText className="mx-auto text-gray-600 mb-4" size={48} />
            <p className="text-gray-400">{t("filesPage.empty.title")}</p>
            <p className="text-gray-500 text-sm mt-1">
              {t("filesPage.empty.desc")}
            </p>
          </div>
        ) : (
          <div className="glass-dark rounded-xl border border-white/10 overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-gray-500 border-b border-white/10">
                  <th className="px-4 py-3 font-medium">{t("filesPage.table.fileName")}</th>
                  <th className="px-4 py-3 font-medium">{t("filesPage.table.size")}</th>
                  <th className="px-4 py-3 font-medium">{t("filesPage.table.date")}</th>
                  <th className="px-4 py-3 font-medium">{t("filesPage.table.type")}</th>
                  <th className="px-4 py-3 font-medium w-24"></th>
                </tr>
              </thead>
              <tbody>
                {files.map((file) => (
                  <tr
                    key={file.id}
                    className="border-b border-white/5 hover:bg-white/5 transition-colors"
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <FileText size={18} className="text-primary-400" />
                        <span className="text-white truncate max-w-[200px]">
                          {file.file_name}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-gray-400 text-sm">
                      {formatSize(file.file_size)}
                    </td>
                    <td className="px-4 py-3 text-gray-400 text-sm">
                      {formatDate(file.created_at, i18n.language)}
                    </td>
                    <td className="px-4 py-3">
                      {file.workflow_type && (
                        <span className="px-2 py-0.5 text-xs rounded-full bg-primary-500/20 text-primary-300">
                          {file.workflow_type}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1">
                        {file.download_url && (
                          <a
                            href={file.download_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="p-1.5 hover:bg-primary-500/20 rounded text-primary-400 transition-colors"
                            title={t("filesPage.actions.download")}
                          >
                            <Download size={16} />
                          </a>
                        )}
                        <button
                          onClick={() => file.id && handleDelete(file.id, file.file_name)}
                          disabled={!file.id || deleting === file.id}
                          className="p-1.5 hover:bg-red-500/20 rounded text-red-400 transition-colors disabled:opacity-50"
                          title={t("filesPage.actions.delete")}
                        >
                          {deleting === file.id ? (
                            <Loader2 size={16} className="animate-spin" />
                          ) : (
                            <Trash2 size={16} />
                          )}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
